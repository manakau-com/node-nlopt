var nlopt = require('./nlopt');
var myfunc = function(n, x, grad){
  if(grad){
    grad[0] = 0.0;
    grad[1] = 0.5 / Math.sqrt(x[1]);
  }
  return Math.sqrt(x[1]);
}
options = {
  algorithm: "LN_COBYLA",
  numberOfParameters:2,
  minObjectiveFunction: myfunc,
  inequalityMConstraints: [{
    callback: function (m, n, x, grad) {
        const cd = [2.0, 0.0, -1.0, 1.0];
        if (grad) {
            grad[0] = 3.0 * cd[0] * (cd[0]*x[0] + cd[1]) * (cd[0]*x[0] + cd[1]);
            grad[1] = -1.0;
            grad[2] = 3.0 * cd[2] * (cd[2]*x[0] + cd[3]) * (cd[2]*x[0] + cd[3]);
            grad[3] = -1.0;
        }
        tmp0 = cd[0]*x[0] + cd[1]
        tmp1 = cd[2]*x[0] + cd[3]
        return [tmp0 * tmp0 * tmp0 - x[1], tmp1 * tmp1 * tmp1 - x[1]];
      },
    tolerances: [1e-8, 1e-8]
  }],
  xToleranceRelative:1e-4,
  initialGuess:[1.234, 5.678],
  lowerBounds:[Number.MIN_VALUE, 0]
}
console.log(nlopt(options).parameterValues);
