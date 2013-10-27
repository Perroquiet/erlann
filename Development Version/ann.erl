%
% ERLANN Library - Development Version (v.0.9.0)
%
-module(ann).
-export([perceptron/5, connect/2, get_config/1, connect_config/6, random_weight/0, expected_vals/1]).


perceptron(Weights, Inputs, Sensitivities, Stale_inputs, Outputs) ->
  Sigmoid = fun(X) -> 1/(1+math:exp(-X)) end,
  Sigmoid_deriv = fun(X) -> math:exp(-X)/(1+math:exp(-2*X)) end,

  receive
    {stimulate, Input} ->
      % add input to inputs
      New_stale_inputs = delete_input(Stale_inputs, Input),
      New_inputs = replace_input(Inputs, Input),
      
      Expected_Val = element(3,Input),
	  case New_stale_inputs of
        % calculate perceptron output
        [] -> Output = feed_forward(Sigmoid, Weights, 
                          convert_to_input_values(New_inputs)),
		
        % stimulate connected perceptrons
        case Sensitivities of
          [] -> 
		  
			case Expected_Val of
				0 ->
					training_list ! {get_list, self()},
					receive
						{ok, ExpectedVal} -> ok
					end;
				ExpectedVal -> ok
			end,
			
			TupOut = lists:keyfind(self(),1,ExpectedVal),
			{_,Eval} = TupOut,
            case Eval of
              null -> 
				io:format("Output (~w): ~w~n", [self(), Output]),
				backdone_pid ! {done_ff};
              E ->
				io:format("Output (~w): ~w~n", [self(), Output]),
				self() ! {learn, {self(), E}}
            end;
		  S -> lists:foreach(fun(Output_PID) ->
                Output_PID ! {stimulate, {self(), Output, Expected_Val}}%, io:format("out: ~w current: ~w Expected: ~w~n", [Output_PID,self(),Expected_Val])
              end,
              convert_to_keys(S))
        end,
        perceptron(Weights, New_inputs, Sensitivities, New_inputs, [Output]);

        New_stale -> perceptron(Weights, New_inputs, Sensitivities, New_stale, Outputs)
      end;

    {learn, Backprop} ->
	
      % {Back_source, Back_value} = Backprop,

      Learning_rate = 0.5,
      New_sensitivities = add_sensitivity(Sensitivities, Backprop),
      Output = feed_forward(Sigmoid, Weights, 
                  convert_to_input_values(Inputs)),
      Deriv = feed_forward(Sigmoid_deriv, Weights, 
                  convert_to_input_values(Inputs)),
      Sensitivity = calculate_sensitivities(Backprop, Inputs,
                  New_sensitivities, Output, Deriv),

      Weight_adjustments = lists:map(fun(Input) ->
                              Learning_rate * Sensitivity * Input
                            end,
                            convert_to_input_values(Inputs)),
      New_weights = vector_map(fun(W, D) -> W+D end, Weights,
                      Weight_adjustments),
      vector_map(fun(Weight, Input_PID) ->
          Input_PID ! {learn, {self(), Sensitivity * Weight}}
        end,
        New_weights,
        convert_to_input_keys(Inputs)),
		
		case Inputs of
			[] ->
				backdone_pid ! {got_answer, self()};
			_ -> ok
		end,
      perceptron(New_weights, Inputs, New_sensitivities, Stale_inputs, Outputs);

    {connect_to_output, Receiver_PID} ->
		Combined_sensitivities = [{Receiver_PID, 0.5} | Sensitivities],
		perceptron(Weights, Inputs, Combined_sensitivities, Stale_inputs, Outputs);

    {connect_to_input, Sender_PID} ->
      Combined_input = [{Sender_PID, 0.5, null} | Inputs],
      New_stale_input = [{Sender_PID, 0.5, null} | Stale_inputs],
      perceptron([random_weight()|Weights], Combined_input, Sensitivities, New_stale_input, Outputs);
	  
	{status, From} ->
		Status = {self(), Weights, Inputs, Sensitivities, Stale_inputs, Outputs},
		From ! {ok, Status},
		perceptron(Weights, Inputs, Sensitivities, Stale_inputs, Outputs);

    {pass, Input_value, Expected_Output} ->
      lists:foreach(fun(Output_PID) ->
		Output_PID ! {stimulate, {self(), Input_value, Expected_Output}}
		end,
        convert_to_keys(Sensitivities)),
      perceptron(Weights, Inputs, Sensitivities, Stale_inputs, Outputs);
	
	{reset} ->
		perceptron(lists:map(fun(_) -> random_weight() end, Weights), Inputs, Sensitivities, Stale_inputs, Outputs);
	
	{return_output, From} ->
		From ! {output, self(), hd(Outputs)},
		perceptron(Weights, Inputs, Sensitivities, Stale_inputs, Outputs);
	
	{get_config, From} ->
		From ! {config, {Weights}},
		perceptron(Weights, Inputs, Sensitivities, Stale_inputs, Outputs);
	
	{load_config, Weight} ->
		perceptron(Weight, Inputs, Sensitivities, Stale_inputs, Outputs)
	
	
	end.

get_config(Pid) ->
	Pid ! {get_config, self()},
	receive
		{config, Config} ->
			Config
	end.
	
expected_vals(Eval) ->
	receive
		{get, From} ->
			From ! {ok, hd(Eval)},
			expected_vals(remove_head(Eval))
	end.
	
remove_head([_|T]) ->
	T.

connect_config(Sender,Receiver, Weights, Inputs, Sensitivities, StaleInputs) ->
	Sender ! {connect_config_output, Receiver, Sensitivities},
	Receiver ! {connect_config_input, Sender, Weights, Inputs, StaleInputs}.  
  
feed_forward(Func, Weights, Inputs) -> 
  Func(dot_prod(Weights, Inputs)).

add_sensitivity([], _Backprop) -> [];
add_sensitivity(Sensitivities, Backprop) -> 
  replace_sensitivity_input(Sensitivities, Backprop).

calculate_sensitivities(_Backprop, [], 
  _Sensitivities, _Output, _Deriv) ->
  null;
calculate_sensitivities({_, Training_value}, _Inputs, 
  [], Output, Deriv) ->
  (Training_value - Output) * Deriv; % (t-z) * f'(net)
calculate_sensitivities(_Backprop, _Inputs, 
  Sensitivities, _Output, Deriv) ->
  Deriv * lists:sum(convert_to_values(Sensitivities)).

connect(Sender, Receiver) ->
  Sender ! {connect_to_output, Receiver},
  Receiver ! {connect_to_input, Sender}.

replace_sensitivity_input(Inputs, Input) ->
  {Input_PID, _} = Input,
  lists:keyreplace(Input_PID, 1, Inputs, Input).

replace_input(Inputs, Input) ->
  {Input_PID, _, _} = Input,
  lists:keyreplace(Input_PID, 1, Inputs, Input).

delete_input(Inputs, Input) ->
  {Input_PID, _, _} = Input,
  lists:keydelete(Input_PID, 1, Inputs).

random_weight() ->
	  random:seed(now()),
		case random:uniform(2) of
			1 -> random:uniform()-1;
			2 -> random:uniform()
		end.

convert_to_input_values(Input_list) ->
  lists:map(fun({_, Val, _}) -> Val end, Input_list).

convert_to_input_keys(Input_list) ->
  lists:map(fun({Key, _, _}) -> Key end, Input_list).

convert_to_values(Tuple_list) ->
  lists:map(fun({_, Val}) -> Val end, Tuple_list).

convert_to_keys(Tuple_list) ->
  lists:map(fun({Key, _}) -> Key end, Tuple_list).

dot_prod(X, Y) -> dot_prod(0, X, Y).

dot_prod(Acc, [], []) -> Acc;
dot_prod(Acc, [X1|X], [Y1|Y]) ->
  dot_prod(X1*Y1 + Acc, X, Y).

vector_map(Func, X, Y) ->
  vector_map([], Func, X, Y).

vector_map(Acc, _Func, [], []) ->
  lists:reverse(Acc);
vector_map(Acc, Func, [Xh | Xt], [Yh | Yt]) ->
  vector_map([Func(Xh, Yh)|Acc], Func, Xt, Yt).
  