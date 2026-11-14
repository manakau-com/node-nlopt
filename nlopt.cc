#include <node.h>
#include <v8.h>
#include <math.h>
#include <nlopt.h>
#include <nan.h>
#include <algorithm>
#include <limits>
#include <list>
#include <string>

using namespace v8;

// One optimize() call. A callback that throws, or a vector constraint's callback
// whose result cannot be read, keeps its error here and stops NLopt, and
// Optimize throws the error once NLopt returns.
struct OptimizationState {
  nlopt_opt opt;
  Global<Value> error;
  // Execution is terminating, so nothing may call into JavaScript again.
  bool terminating = false;

  bool failed() const { return terminating || !error.IsEmpty(); }
};

// What NLopt hands back to a callback: the JavaScript function, and the
// optimize() call it belongs to.
struct Callback {
  Function* callback;
  OptimizationState* state;
};

// Keeps what tryCatch caught, or notes that execution is terminating, and
// stops NLopt.
void stopOptimization(Isolate* isolate, OptimizationState* state, const TryCatch& tryCatch) {
  if (tryCatch.CanContinue()) {
    state->error.Reset(isolate, tryCatch.Exception());
  }
  else {
    state->terminating = true;
  }
  nlopt_force_stop(state->opt);
}

Local<Array> cArrayToV8Array(unsigned n, const double* array) {
  Isolate* isolate = Isolate::GetCurrent();
  Local<Context> context = isolate->GetCurrentContext();
  Local<Array> ret = Array::New(isolate, n);

  for (unsigned i = 0; i < n; ++i) {
    ret->Set(context, i, Number::New(isolate, array[i])).FromJust();
  }

  return ret;
}

double* v8ArrayToCArray(Local<Array>& array) {
  Isolate* isolate = Isolate::GetCurrent();
  Local<Context> context = isolate->GetCurrentContext();
  double* ret = new double[array->Length()];

  for (unsigned i = 0; i < array->Length(); ++i) {
    Local<Value> val = array->Get(context, i).ToLocalChecked();
    ret[i] = val->NumberValue(context).FromJust();
  }

  return ret;
}

void checkNloptErrorCode(Local<Object>& errors, Local<String>& operation, nlopt_result errorCode) {
  Isolate* isolate = Isolate::GetCurrent();
  Local<Context> context = isolate->GetCurrentContext();
  const char* str;

  switch (errorCode) {
    case NLOPT_SUCCESS:
      str = "Success";
      break;
    case NLOPT_STOPVAL_REACHED:
      str = "Success: Optimization stopped because stopValue was reached";
      break;
    case NLOPT_FTOL_REACHED:
      str = "Success: Optimization stopped because fToleranceRelative or fToleranceAbsolute was reached";
      break;
    case NLOPT_XTOL_REACHED:
      str = "Success: Optimization stopped because xToleranceRelative or xToleranceAbsolute was reached";
      break;
    case NLOPT_MAXEVAL_REACHED:
      str = "Success: Optimization stopped because maxEval was reached";
      break;
    case NLOPT_MAXTIME_REACHED:
      str = "Success: Optimization stopped because maxTime was reached";
      break;
    case NLOPT_FAILURE:
      str = "Failure";
      break;
    case NLOPT_INVALID_ARGS:
      str = "Failure: Invalid arguments";
      break;
    case NLOPT_OUT_OF_MEMORY:
      str = "Failure: Ran out of memory";
      break;
    case NLOPT_ROUNDOFF_LIMITED:
      str = "Failure: Halted because roundoff errors limited progress";
      break;
    case NLOPT_FORCED_STOP:
      str = "Failure: Halted because of a forced termination";
      break;
    default:
      str = "Failure: Unknown Error Code";
      break;
  }

  errors->Set(context, operation, String::NewFromUtf8(isolate, str).ToLocalChecked()).FromJust();
}

#define GET_VALUE(TYPE, NAME, OBJ) \
  Local<String> key_##NAME = String::NewFromUtf8(isolate, #NAME).ToLocalChecked(); \
  Local<TYPE> val_##NAME; \
  if ((OBJ)->Has(context, key_##NAME).FromJust()) { \
    val_##NAME = Local<TYPE>::Cast((OBJ)->Get(context, key_##NAME).ToLocalChecked()); \
  }

#define CHECK_CODE(NAME) \
  checkNloptErrorCode(ret, key_##NAME, code);

#define SIMPLE_CONFIG_OPTION(NAME, CONFIG_METHOD) \
  GET_VALUE(Number, NAME, options) \
  if(!val_##NAME.IsEmpty()){ \
    code = CONFIG_METHOD(opt, val_##NAME->Value()); \
    CHECK_CODE(NAME) \
  }

double optimizationFunc(unsigned n, const double* x, double* grad, void* data)
{
  Callback* function = static_cast<Callback*>(data);
  OptimizationState* state = function->state;
  if (state->failed()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  Isolate* isolate = Isolate::GetCurrent();
  HandleScope scope(isolate);
  Local<Context> context = isolate->GetCurrentContext();

  double returnValue = -1;

  //prepare parms to callback
  Local<Value> argv[3];
  argv[0] = Number::New(isolate, n);
  argv[1] = cArrayToV8Array(n, x);
  //gradient
  Local<Array> v8Grad;
  if(grad){
    v8Grad = cArrayToV8Array(n, grad);
    argv[2] = v8Grad;
  }
  else {
    argv[2] = Null(isolate);
  }
  // Call callback 
  TryCatch tryCatch(isolate);
  Local<Value> ret;
  if (!function->callback->Call(context, context->Global(), 3, argv).ToLocal(&ret)) {
    stopOptimization(isolate, state, tryCatch);
    return std::numeric_limits<double>::quiet_NaN();
  }
  //validate return results
  if(!ret->IsNumber()){
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Objective or constraint function must return a number.").ToLocalChecked()));
  }
  else if(grad && v8Grad->Length() != n){
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Length of gradient array must be the same as the number of parameters.").ToLocalChecked()));
  }
  else { //success
    if(grad){
      for (unsigned i = 0; i < n; ++i) {
        grad[i] = v8Grad->Get(context, i).ToLocalChecked()->NumberValue(context).ToChecked();
      }
    }
    returnValue = ret->NumberValue(context).ToChecked();
  }
  return returnValue;
}

void throwTypeError(Isolate* isolate, const std::string& message) {
  isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, message.c_str()).ToLocalChecked()));
}

// Copies the numbers of a plain array into values. False, with an exception
// thrown, if reading an element throws.
bool copyArray(Local<Context> context, Local<Array> array, unsigned length, double* values) {
  for (unsigned i = 0; i < length; ++i) {
    Local<Value> element;
    if (!array->Get(context, i).ToLocal(&element)) {
      return false;
    }
    Maybe<double> number = element->NumberValue(context);
    if (number.IsNothing()) {
      return false;
    }
    values[i] = number.FromJust();
  }
  return true;
}

bool checkMConstraintLength(Isolate* isolate, size_t length, unsigned m) {
  if (length != m) {
    throwTypeError(isolate, "A vector constraint's callback must return " + std::to_string(m) +
      " values, one for each tolerance, not " + std::to_string(length) + ".");
    return false;
  }
  return true;
}

// A vector constraint's callback returns its m values as a plain array or a
// Float64Array. False, with an exception thrown, for anything else.
bool readMConstraintResult(Isolate* isolate, Local<Context> context, Local<Value> ret, unsigned m, double* result) {
  if (ret->IsFloat64Array()) {
    Local<Float64Array> values = ret.As<Float64Array>();
    if (!checkMConstraintLength(isolate, values->Length(), m)) {
      return false;
    }
    values->CopyContents(result, m * sizeof(double));
    return true;
  }
  if (ret->IsArray()) {
    Local<Array> values = ret.As<Array>();
    return checkMConstraintLength(isolate, values->Length(), m) && copyArray(context, values, m, result);
  }
  String::Utf8Value type(isolate, ret->TypeOf(isolate));
  throwTypeError(isolate, std::string("A vector constraint's callback must return an array or a Float64Array of numbers, not ") + *type + ".");
  return false;
}

void optimizationMFunc(unsigned m, double* result, unsigned n, const double* x, double* grad, void* data)
{
  Callback* constraint = static_cast<Callback*>(data);
  OptimizationState* state = constraint->state;
  std::fill(result, result + m, std::numeric_limits<double>::quiet_NaN());
  if (state->failed()) {
    return;
  }

  Isolate* isolate = Isolate::GetCurrent();
  HandleScope scope(isolate);
  Local<Context> context = isolate->GetCurrentContext();

  //prepare parms to callback
  Local<Value> argv[4];
  argv[0] = Number::New(isolate, m);
  argv[1] = Number::New(isolate, n);
  argv[2] = cArrayToV8Array(n, x);
  // NLopt's gradient for m constraints is m rows of n.
  unsigned gradLength = m * n;
  Local<Array> v8Grad;
  if(grad){
    v8Grad = cArrayToV8Array(gradLength, grad);
    argv[3] = v8Grad;
  }
  else {
    argv[3] = Null(isolate);
  }

  TryCatch tryCatch(isolate);
  Local<Value> ret;
  bool ok = constraint->callback->Call(context, context->Global(), 4, argv).ToLocal(&ret) &&
    readMConstraintResult(isolate, context, ret, m, result);
  if (ok && grad) {
    if (v8Grad->Length() != gradLength) {
      throwTypeError(isolate, "A vector constraint's gradient must have " + std::to_string(gradLength) +
        " entries, the number of constraints times the number of parameters, not " + std::to_string(v8Grad->Length()) + ".");
      ok = false;
    }
    else {
      ok = copyArray(context, v8Grad, gradLength, grad);
    }
  }
  if (!ok) {
    stopOptimization(isolate, state, tryCatch);
  }
}

bool hasValue(const Local<Value>& v) {
  return !v.IsEmpty() && !v->IsUndefined() && !v->IsNull();
}

NAN_METHOD(Optimize) {
  Isolate* isolate = Isolate::GetCurrent();
  EscapableHandleScope scope(isolate);
  Local<Context> context = isolate->GetCurrentContext();

  Local<Object> ret = Object::New(isolate);
  nlopt_result code = NLOPT_SUCCESS;
  Local<String> key;

  // There is not much validation in this function... should be done in JS.
  Local<Object> options = info[0].As<Object>();

  // Basic NLOpt config
  GET_VALUE(Number, algorithm, options)
  GET_VALUE(Number, numberOfParameters, options)
  unsigned n = val_numberOfParameters->Uint32Value(context).FromJust();
  nlopt_opt opt = nlopt_create(static_cast<nlopt_algorithm>(val_algorithm->Uint32Value(context).FromJust()), n);
  OptimizationState state;
  state.opt = opt;
  std::list<Callback> callbacks;

  // Objective function
  GET_VALUE(Function, minObjectiveFunction, options)
  GET_VALUE(Function, maxObjectiveFunction, options)
  int minMax = 0;
  if (hasValue(val_minObjectiveFunction)) {
    callbacks.push_back({*val_minObjectiveFunction, &state});
    code = nlopt_set_min_objective(opt, optimizationFunc, &callbacks.back());
    CHECK_CODE(minObjectiveFunction)
    ++minMax;
  }
  if (hasValue(val_maxObjectiveFunction)) {
    callbacks.push_back({*val_maxObjectiveFunction, &state});
    code = nlopt_set_max_objective(opt, optimizationFunc, &callbacks.back());
    CHECK_CODE(maxObjectiveFunction)
    ++minMax;
  }
  if (minMax != 1) {
    isolate->ThrowException(Exception::TypeError(
      String::NewFromUtf8(isolate, "minObjectiveFunction or maxObjectiveFunction must be specified").ToLocalChecked()
    ));
    info.GetReturnValue().Set(scope.Escape(ret));
    return;
  }

  // Optional parameters
  GET_VALUE(Array, lowerBounds, options)
  if (hasValue(val_lowerBounds)) {
    double* lowerBounds = v8ArrayToCArray(val_lowerBounds);
    code = nlopt_set_lower_bounds(opt, lowerBounds);
    CHECK_CODE(lowerBounds)
    delete[] lowerBounds;
  }

  GET_VALUE(Array, upperBounds, options)
  if (hasValue(val_upperBounds)) {
    double* upperBounds = v8ArrayToCArray(val_upperBounds);
    code = nlopt_set_upper_bounds(opt, upperBounds);
    CHECK_CODE(upperBounds)
    delete[] upperBounds;
  }

  SIMPLE_CONFIG_OPTION(stopValue, nlopt_set_stopval)
  SIMPLE_CONFIG_OPTION(fToleranceRelative, nlopt_set_ftol_rel)
  SIMPLE_CONFIG_OPTION(fToleranceAbsolute, nlopt_set_ftol_abs)
  SIMPLE_CONFIG_OPTION(xToleranceRelative, nlopt_set_xtol_rel)
  SIMPLE_CONFIG_OPTION(xToleranceAbsolute, nlopt_set_xtol_abs1)
  SIMPLE_CONFIG_OPTION(maxEval, nlopt_set_maxeval)
  SIMPLE_CONFIG_OPTION(maxTime, nlopt_set_maxtime)

  GET_VALUE(Array, inequalityConstraints, options)
  if (hasValue(val_inequalityConstraints)) {
    for (unsigned i = 0; i < val_inequalityConstraints->Length(); ++i) {
      Local<Object> obj = val_inequalityConstraints->Get(context, i).ToLocalChecked().As<Object>();
      GET_VALUE(Function, callback, obj)
      GET_VALUE(Number, tolerance, obj)
      callbacks.push_back({*val_callback, &state});
      code = nlopt_add_inequality_constraint(opt, optimizationFunc, &callbacks.back(), val_tolerance->NumberValue(context).FromJust());
      CHECK_CODE(inequalityConstraints)
    }
  }

  GET_VALUE(Array, equalityConstraints, options)
  if (hasValue(val_equalityConstraints)) {
    for (unsigned i = 0; i < val_equalityConstraints->Length(); ++i) {
      Local<Object> obj = val_equalityConstraints->Get(context, i).ToLocalChecked().As<Object>();
      GET_VALUE(Function, callback, obj)
      GET_VALUE(Number, tolerance, obj)
      callbacks.push_back({*val_callback, &state});
      code = nlopt_add_equality_constraint(opt, optimizationFunc, &callbacks.back(), val_tolerance->NumberValue(context).FromJust());
      CHECK_CODE(equalityConstraints)
    }
  }

  GET_VALUE(Array, inequalityMConstraints, options)
  if (hasValue(val_inequalityMConstraints)) {
    for (unsigned i = 0; i < val_inequalityMConstraints->Length(); ++i) {
      Local<Object> obj = val_inequalityMConstraints->Get(context, i).ToLocalChecked().As<Object>();
      GET_VALUE(Function, callback, obj)
      GET_VALUE(Array, tolerances, obj)
      double* tolerances = v8ArrayToCArray(val_tolerances);
      callbacks.push_back({*val_callback, &state});
      code = nlopt_add_inequality_mconstraint(opt, val_tolerances->Length(), optimizationMFunc, &callbacks.back(), tolerances);
      CHECK_CODE(inequalityMConstraints)
    }
  }

  GET_VALUE(Array, equalityMConstraints, options)
  if (hasValue(val_equalityMConstraints)) {
    for (unsigned i = 0; i < val_equalityMConstraints->Length(); ++i) {
      Local<Object> obj = val_equalityMConstraints->Get(context, i).ToLocalChecked().As<Object>();
      GET_VALUE(Function, callback, obj)
      GET_VALUE(Array, tolerances, obj)
      double* tolerances = v8ArrayToCArray(val_tolerances);
      callbacks.push_back({*val_callback, &state});
      code = nlopt_add_equality_mconstraint(opt, val_tolerances->Length(), optimizationMFunc, &callbacks.back(), tolerances);
      CHECK_CODE(equalityMConstraints)
    }
  }

  // Setup parameters for optimization
  double* input = new double[n];
  std::fill(input, input + n, 0);

  // Initial guess
  GET_VALUE(Array, initialGuess, options)
  if (hasValue(val_initialGuess)) {
    ret->Set(context, key_initialGuess, String::NewFromUtf8(isolate, "Success").ToLocalChecked()).FromJust();
    for (unsigned i = 0; i < val_initialGuess->Length(); ++i) {
      input[i] = val_initialGuess->Get(context, i).ToLocalChecked()->NumberValue(context).FromJust();
    }
  }

  // Do the optimization!
  key = String::NewFromUtf8(isolate, "status").ToLocalChecked();
  double output[1] = {0};
  nlopt_result status = nlopt_optimize(opt, input, output);
  if (state.failed()) {
    delete[] input;
    nlopt_destroy(opt);
    if (!state.terminating) {
      isolate->ThrowException(state.error.Get(isolate));
    }
    return;
  }
  checkNloptErrorCode(ret, key, status);
  ret->Set(context, String::NewFromUtf8(isolate, "parameterValues").ToLocalChecked(), cArrayToV8Array(n, input)).FromJust();
  delete[] input;
  ret->Set(context, String::NewFromUtf8(isolate, "outputValue").ToLocalChecked(), Number::New(isolate, output[0])).FromJust();
  nlopt_destroy(opt); // Cleanup
  info.GetReturnValue().Set(scope.Escape(ret));
}

NAN_MODULE_INIT(init) {
  Nan::Export(target, "optimize", Optimize);
}

NAN_MODULE_WORKER_ENABLED(nlopt, init)
