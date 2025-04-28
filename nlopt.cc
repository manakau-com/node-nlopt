#include <node.h>
#include <v8.h>
#include <math.h>
#include <nlopt.h>
#include <nan.h>

using namespace v8;

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

double optimizationFunc(unsigned n, const double* x, double* grad, void* ptrCallback)
{
  Isolate* isolate = Isolate::GetCurrent();
  EscapableHandleScope scope(isolate);
  Local<Context> context = isolate->GetCurrentContext();

  Local<Value> undefined;
  //Local<Function> callback = *reinterpret_cast<Local<Function>*>(ptrCallback);
  Function* callback = (Function*)(ptrCallback);
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
  auto ret = callback->Call(context, context->Global(), 3, argv).ToLocalChecked();
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
  scope.Escape(undefined);
  return returnValue;
}

void optimizationMFunc(unsigned m, double* result, unsigned n, const double* x, double* grad, void* ptrCallback)
{
  Isolate* isolate = Isolate::GetCurrent();
  EscapableHandleScope scope(isolate);
  Local<Context> context = isolate->GetCurrentContext();

  Local<Value> undefined;
  //Local<Function> callback = *reinterpret_cast<Local<Function>*>(ptrCallback);
  Function* callback = (Function*)(ptrCallback);

  //prepare parms to callback
  Local<Value> argv[4];
  argv[0] = Number::New(isolate, m);
  argv[1] = Number::New(isolate, n);
  argv[2] = cArrayToV8Array(n, x);
  //gradient
  Local<Array> v8Grad;
  if(grad){
    v8Grad = cArrayToV8Array(n, grad);
    argv[3] = v8Grad;
  }
  else {
    argv[3] = Null(isolate);
  }
  // Call callback 
  TryCatch tryCatch(isolate);
  auto ret = callback->Call(context, context->Global(), 4, argv).ToLocalChecked();
  //validate return results
  if(!ret->IsFloat64Array()){
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Objective or constraint function must return an array of number.").ToLocalChecked()));
  }

  Local<Array> resultArray = Local<Array>::Cast(ret);
  if(resultArray->Length() != m){
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Length of result array must be the same as the m parameter.").ToLocalChecked()));
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
    for (unsigned i = 0; i < m; ++i) {
      result[i] = resultArray->Get(context, i).ToLocalChecked()->NumberValue(context).ToChecked();
    }
  }
  scope.Escape(undefined);
}

bool hasValue(const Local<Value>& v) {
  return !v.IsEmpty() && !v->IsUndefined() && !v->IsNull();
}

// void Optimize(const v8::FunctionCallbackInfo<v8::Value>& args) {
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

  // Objective function
  GET_VALUE(Function, minObjectiveFunction, options)
  GET_VALUE(Function, maxObjectiveFunction, options)
  int minMax = 0;
  if (hasValue(val_minObjectiveFunction)) {
    code = nlopt_set_min_objective(opt, optimizationFunc, *val_minObjectiveFunction);
    CHECK_CODE(minObjectiveFunction)
    ++minMax;
  }
  if (hasValue(val_maxObjectiveFunction)) {
    code = nlopt_set_max_objective(opt, optimizationFunc, *val_maxObjectiveFunction);
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
      code = nlopt_add_inequality_constraint(opt, optimizationFunc, *val_callback, val_tolerance->NumberValue(context).FromJust());
      CHECK_CODE(inequalityConstraints)
    }
  }

  GET_VALUE(Array, equalityConstraints, options)
  if (hasValue(val_equalityConstraints)) {
    for (unsigned i = 0; i < val_equalityConstraints->Length(); ++i) {
      Local<Object> obj = val_equalityConstraints->Get(context, i).ToLocalChecked().As<Object>();
      GET_VALUE(Function, callback, obj)
      GET_VALUE(Number, tolerance, obj)
      code = nlopt_add_equality_constraint(opt, optimizationFunc, *val_callback, val_tolerance->NumberValue(context).FromJust());
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
      code = nlopt_add_inequality_mconstraint(opt, val_tolerances->Length(), optimizationMFunc, *val_callback, tolerances);
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
      code = nlopt_add_equality_mconstraint(opt, val_tolerances->Length(), optimizationMFunc, *val_callback, tolerances);
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
  checkNloptErrorCode(ret, key, nlopt_optimize(opt, input, output));
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
