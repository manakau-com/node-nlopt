var childProcess = require('child_process');
var path = require('path');
var expect = require('expect.js');
var nlopt = require('../nlopt.js');

// Minimise (x - 0.3)^2 for x in [0, 1], optionally with a scalar constraint.
function problem(objective, inequalityConstraints) {
  return {
    algorithm: 'LN_COBYLA',
    numberOfParameters: 1,
    minObjectiveFunction: objective,
    inequalityConstraints: inequalityConstraints,
    lowerBounds: [0],
    upperBounds: [1],
    initialGuess: [0.9],
    xToleranceRelative: 1e-8,
    maxEval: 1000
  };
}

function square(n, x) {
  return (x[0] - 0.3) * (x[0] - 0.3);
}

describe('the objective and scalar constraints', function () {
  it('throws the error the objective throws, and calls it no more', function () {
    var thrown = new Error('objective failed');
    var calls = 0;
    var options = problem(function (n, x) {
      if (++calls === 5) {
        throw thrown;
      }
      return square(n, x);
    });
    expect(function () { nlopt(options); }).to.throwException(function (e) {
      expect(e).to.be(thrown);
    });
    expect(calls).to.be(5);
  });

  it('throws the error a scalar constraint throws, and calls nothing more', function () {
    var thrown = new Error('constraint failed');
    var objectiveCalls = 0;
    var constraintCalls = 0;
    var options = problem(function (n, x) {
      ++objectiveCalls;
      return square(n, x);
    }, [{
      callback: function (n, x) {
        if (++constraintCalls === 5) {
          throw thrown;
        }
        return x[0] - 2;
      },
      tolerance: 0
    }]);
    expect(function () { nlopt(options); }).to.throwException(function (e) {
      expect(e).to.be(thrown);
    });
    expect(constraintCalls).to.be(5);
    expect(objectiveCalls).to.be(5);
  });

  it('still answers a solve whose callbacks do not throw', function () {
    var result = nlopt(problem(square, [{ callback: function (n, x) { return x[0] - 2; }, tolerance: 0 }]));
    expect(result.status).to.match(/^Success/);
    expect(Math.abs(result.parameterValues[0] - 0.3)).to.be.below(1e-6);
  });

  it('lets a worker be terminated inside the objective', function () {
    // In a child process, so that an abort fails this test rather than the run.
    var script = [
      "const { Worker } = require('worker_threads');",
      'const worker = new Worker(' + JSON.stringify([
        "const { parentPort } = require('worker_threads');",
        'const nlopt = require(' + JSON.stringify(path.join(__dirname, '..', 'nlopt.js')) + ');',
        'let calls = 0;',
        'nlopt({',
        "  algorithm: 'LN_COBYLA', numberOfParameters: 1,",
        '  minObjectiveFunction: (n, x) => {',
        "    if (++calls === 5) { parentPort.postMessage('spinning'); for (;;) {} }",
        '    return x[0] * x[0];',
        '  },',
        '  inequalityMConstraints: [{ tolerances: [0], callback: () => [0] }],',
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
