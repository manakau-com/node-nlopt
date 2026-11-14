var childProcess = require('child_process');
var path = require('path');
var expect = require('expect.js');
var nlopt = require('../nlopt.js');

// Four weights between 0 and 1 that sum to 1, the first two summing to 0.4,
// as one vector equality constraint of two values. The objective alone would
// put everything on the last weight, so the answer depends on NLopt reading
// both constraint values.
function portfolio(constraintResult) {
  var returns = [0.02, 0.03, 0.04, 0.05];
  var vols = [0.1, 0.15, 0.2, 0.25];
  return {
    algorithm: 'LN_COBYLA',
    numberOfParameters: 4,
    maxObjectiveFunction: function (n, w) {
      var value = 0;
      for (var i = 0; i < n; i++) {
        value += w[i] * returns[i] - 0.5 * w[i] * w[i] * vols[i] * vols[i];
      }
      return value;
    },
    equalityMConstraints: [{
      callback: function (m, n, w) {
        return constraintResult(w[0] + w[1] + w[2] + w[3] - 1, w[0] + w[1] - 0.4);
      },
      tolerances: [1e-8, 1e-8]
    }],
    lowerBounds: [0, 0, 0, 0],
    upperBounds: [1, 1, 1, 1],
    initialGuess: [0.25, 0.25, 0.25, 0.25],
    xToleranceRelative: 1e-6,
    maxEval: 10000
  };
}

function expectConstraintsHold(result) {
  var w = result.parameterValues;
  expect(result.status).to.match(/^Success/);
  expect(Math.abs(w[0] + w[1] + w[2] + w[3] - 1)).to.be.below(1e-6);
  expect(Math.abs(w[0] + w[1] - 0.4)).to.be.below(1e-6);
}

function expectTypeError(options, pattern) {
  expect(function () { nlopt(options); }).to.throwException(function (e) {
    expect(e).to.be.a(TypeError);
    expect(e.message).to.match(pattern);
  });
}

describe('vector constraints', function () {
  it('reads a plain array', function () {
    expectConstraintsHold(nlopt(portfolio(function (a, b) { return [a, b]; })));
  });

  it('reads a Float64Array, with the same answer as a plain array', function () {
    var plain = nlopt(portfolio(function (a, b) { return [a, b]; }));
    var typed = nlopt(portfolio(function (a, b) { return Float64Array.of(a, b); }));
    expectConstraintsHold(typed);
    expect(typed.parameterValues).to.eql(plain.parameterValues);
    expect(typed.outputValue).to.be(plain.outputValue);
  });

  it('throws when the callback returns the wrong number of values', function () {
    var message = /must return 2 values, one for each tolerance, not (1|3)\./;
    expectTypeError(portfolio(function (a) { return [a]; }), message);
    expectTypeError(portfolio(function (a, b) { return Float64Array.of(a, b, 0); }), message);
  });

  it('throws when the callback returns something other than an array', function () {
    var message = /must return an array or a Float64Array of numbers, not (number|object|undefined)\./;
    expectTypeError(portfolio(function (a) { return a; }), message);
    expectTypeError(portfolio(function (a, b) { return { 0: a, 1: b, length: 2 }; }), message);
    expectTypeError(portfolio(function () { return undefined; }), message);
  });

  it('throws the error the callback throws', function () {
    var thrown = new Error('constraint failed');
    var calls = 0;
    var options = portfolio(function () {
      if (++calls === 5) {
        throw thrown;
      }
      return [0, 0];
    });
    expect(function () { nlopt(options); }).to.throwException(function (e) {
      expect(e).to.be(thrown);
    });
    expect(calls).to.be(5);

    var unreadable = portfolio(function () {
      return [0, { valueOf: function () { throw thrown; } }];
    });
    expect(function () { nlopt(unreadable); }).to.throwException(function (e) {
      expect(e).to.be(thrown);
    });
  });

  it('passes the gradient as one row of parameters per constraint', function () {
    // NLopt's tutorial problem, with its two constraints as one vector.
    var result = nlopt({
      algorithm: 'LD_MMA',
      numberOfParameters: 2,
      minObjectiveFunction: function (n, x, grad) {
        if (grad) {
          grad[0] = 0;
          grad[1] = 0.5 / Math.sqrt(x[1]);
        }
        return Math.sqrt(x[1]);
      },
      inequalityMConstraints: [{
        callback: function (m, n, x, grad) {
          var a = [2, -1];
          var b = [0, 1];
          var values = [];
          for (var i = 0; i < m; i++) {
            var t = a[i] * x[0] + b[i];
            if (grad) {
              grad[i * n] = 3 * a[i] * t * t;
              grad[i * n + 1] = -1;
            }
            values.push(t * t * t - x[1]);
          }
          return values;
        },
        tolerances: [1e-8, 1e-8]
      }],
      lowerBounds: [-Infinity, 0],
      initialGuess: [1.234, 5.678],
      xToleranceRelative: 1e-4
    });
    expect(result.status).to.match(/^Success/);
    expect(Math.abs(result.parameterValues[0] - 1 / 3)).to.be.below(1e-4);
    expect(Math.abs(result.parameterValues[1] - 8 / 27)).to.be.below(1e-4);
  });

  it('throws when the callback changes the length of the gradient', function () {
    expectTypeError({
      algorithm: 'LD_MMA',
      numberOfParameters: 2,
      minObjectiveFunction: function (n, x, grad) {
        if (grad) {
          grad[0] = 0;
          grad[1] = 1;
        }
        return x[1];
      },
      inequalityMConstraints: [{
        callback: function (m, n, x, grad) {
          if (grad) {
            grad.push(0);
          }
          return [x[0] - 1, -x[1]];
        },
        tolerances: [0, 0]
      }],
      initialGuess: [0, 1],
      maxEval: 100
    }, /gradient must have 4 entries, the number of constraints times the number of parameters, not 5\./);
  });

  it('lets a worker be terminated inside the callback', function () {
    // In a child process, so that an abort fails this test rather than the run.
    var script = [
      "const { Worker } = require('worker_threads');",
      'const worker = new Worker(' + JSON.stringify([
        "const { parentPort } = require('worker_threads');",
        'const nlopt = require(' + JSON.stringify(path.join(__dirname, '..', 'nlopt.js')) + ');',
        'let calls = 0;',
        'nlopt({',
        "  algorithm: 'LN_COBYLA', numberOfParameters: 1,",
        '  minObjectiveFunction: (n, x) => x[0] * x[0],',
        '  inequalityMConstraints: [{ tolerances: [0], callback: () => {',
        "    if (++calls === 5) { parentPort.postMessage('spinning'); for (;;) {} }",
        '    return [0];',
        '  } }],',
        '  initialGuess: [1], maxEval: 1000',
        '});'
      ].join('\n')) + ', { eval: true });',
      "worker.on('message', () => worker.terminate());",
      "worker.on('exit', (code) => console.log('worker exit ' + code));"
    ].join('\n');
    var child = childProcess.spawnSync(process.execPath, ['-e', script], { encoding: 'utf8', timeout: 20000 });
    expect(child.stderr).to.be('');
    expect(child.status).to.be(0);
    expect(child.stdout).to.be('worker exit 1\n');
  });
});
