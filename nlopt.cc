#include <node.h>
#include <v8.h>
#include <math.h>
#include <nlopt.h>
#include <stdio.h>

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

void Optimize(const v8::FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = Isolate::GetCurrent();
  EscapableHandleScope scope(isolate);
  Local<Context> context = isolate->GetCurrentContext();

  Local<Object> ret = Object::New(isolate);
  nlopt_result code = NLOPT_SUCCESS;
  Local<String> key;

  // There is not much validation in this function... should be done in JS.
  Local<Object> options = args[0].As<Object>();

  // Basic NLOpt config
  GET_VALUE(Number, algorithm, options)
  GET_VALUE(Number, numberOfParameters, options)
  unsigned n = val_numberOfParameters->Uint32Value(context).FromJust();
  nlopt_opt opt = nlopt_create(static_cast<nlopt_algorithm>(val_algorithm->Uint32Value(context).FromJust()), n);

  // Objective function
  GET_VALUE(Function, minObjectiveFunction, options)
  GET_VALUE(Function, maxObjectiveFunction, options)
  if (!val_minObjectiveFunction.IsEmpty()) {
    code = nlopt_set_min_objective(opt, optimizationFunc, *val_minObjectiveFunction);
    CHECK_CODE(minObjectiveFunction)
  } else if (!val_maxObjectiveFunction.IsEmpty()) {
    code = nlopt_set_max_objective(opt, optimizationFunc, *val_maxObjectiveFunction);
    CHECK_CODE(maxObjectiveFunction)
  } else {
    isolate->ThrowException(Exception::TypeError(
      String::NewFromUtf8(isolate, "minObjectiveFunction or maxObjectiveFunction must be specified").ToLocalChecked()
    ));
    args.GetReturnValue().Set(scope.Escape(ret));
    return;
  }

  // Optional parameters
  GET_VALUE(Array, lowerBounds, options)
  if (!val_lowerBounds.IsEmpty()) {
    double* lowerBounds = v8ArrayToCArray(val_lowerBounds);
    code = nlopt_set_lower_bounds(opt, lowerBounds);
    CHECK_CODE(lowerBounds)
    delete[] lowerBounds;
  }

  GET_VALUE(Array, upperBounds, options)
  if (!val_upperBounds.IsEmpty()) {
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
  if (!val_inequalityConstraints.IsEmpty()) {
    for (unsigned i = 0; i < val_inequalityConstraints->Length(); ++i) {
      Local<Object> obj = val_inequalityConstraints->Get(context, i).ToLocalChecked().As<Object>();
      GET_VALUE(Function, callback, obj)
      GET_VALUE(Number, tolerance, obj)
      code = nlopt_add_inequality_constraint(opt, optimizationFunc, *val_callback, val_tolerance->NumberValue(context).FromJust());
      CHECK_CODE(inequalityConstraints)
    }
  }

  GET_VALUE(Array, equalityConstraints, options)
  if (!val_equalityConstraints.IsEmpty()) {
    for (unsigned i = 0; i < val_equalityConstraints->Length(); ++i) {
      Local<Object> obj = val_equalityConstraints->Get(context, i).ToLocalChecked().As<Object>();
      GET_VALUE(Function, callback, obj)
      GET_VALUE(Number, tolerance, obj)
      code = nlopt_add_equality_constraint(opt, optimizationFunc, *val_callback, val_tolerance->NumberValue(context).FromJust());
      CHECK_CODE(equalityConstraints)
    }
  }

  // Setup parameters for optimization
  double* input = new double[n];
  std::fill(input, input + n, 0);

  // Initial guess
  GET_VALUE(Array, initalGuess, options)
  if (!val_initalGuess.IsEmpty()) {
    ret->Set(context, key_initalGuess, String::NewFromUtf8(isolate, "Success").ToLocalChecked()).FromJust();
    for (unsigned i = 0; i < val_initalGuess->Length(); ++i) {
      input[i] = val_initalGuess->Get(context, i).ToLocalChecked()->NumberValue(context).FromJust();
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
  args.GetReturnValue().Set(scope.Escape(ret));
}

void init(Local<Object> exports, Local<Value> module, void* priv) {
  Isolate* isolate = Isolate::GetCurrent();
  Local<Context> context = isolate->GetCurrentContext();

  exports->Set(
    context,
    String::NewFromUtf8(isolate, "optimize").ToLocalChecked(),
    FunctionTemplate::New(isolate, Optimize)->GetFunction(context).ToLocalChecked()
  ).FromJust();
}

//NODE_MODULE(NODE_GYP_MODULE_NAME, init)

NODE_MODULE(nlopt, init)
