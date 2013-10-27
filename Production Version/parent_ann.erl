%
% ERLANN Library - Production Version (v.0.9.0)
%
-module(parent_ann).
-export([loop/7]).

%
% Parent Process Loop - This is where the state of the neural network is saved.
%
loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError) ->
receive
	{iLayer_Pid, Pid, Index} ->
		Pid ! {ok,get_pid(InputLayer,Index)},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
		
	{hLayer_Pid, Pid, Index, InnerIndex} ->	
		Pid ! {ok,get_HL_pid(HiddenLayer, Index, InnerIndex)},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
		
	{oLayer_Pid, Pid, Index} ->
		Pid ! {ok,get_pid(OutputLayer,Index)},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
		
	{get_input_layer, Pid} ->
		Pid ! {input_layer, InputLayer},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
	
	{get_hidden_layer, Pid} ->
		Pid ! {hidden_layer, HiddenLayer},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
	
	{get_output_layer, Pid} ->
		Pid ! {output_layer, OutputLayer},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
		
	{set_norm, DenormFactor, Exp, LogExp} ->
		loop(InputLayer, HiddenLayer, OutputLayer, [{DenormFactor,Exp,LogExp}], TrainingPoints, TestingError, MSPError);
		
	{get_norm, From} ->
		From ! {ok, hd(Factors)},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
		
	{get_expected, From} ->
		P = hd(Factors),
		{_,ExpectedVal,_} = P,
		From ! {ok, ExpectedVal},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
		
	{get_log_expected, From} ->
		P = hd(Factors),
		{_,_,LogExpectedVal} = P,
		From ! {ok, LogExpectedVal},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
	
	{return_status, NodePid, From} ->
		NodePid ! {status, self()},
		receive
			{ok, Status} ->
				From ! {ok, Status}	
		end,
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
	
	{return_output, NodePid, From} ->
		NodePid ! {return_output, self()},
		receive
			{output, Pid, Output} ->
				From ! {ok, Pid, Output}
		end,
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
	
	{pass, NodePid, InputValue, Expected_Val} ->
		NodePid ! {pass, InputValue, Expected_Val},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);

	{set_training_points, Tp} ->
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, Tp, TestingError, MSPError);
		
	{get_training_point, From} ->
		[H|T] = TrainingPoints,
		From ! {ok, H},
		io:format("Data Left: ~w~n", [length(TrainingPoints)]),
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, T, TestingError, MSPError);
	
	{count_training_set, From} ->
		From ! {ok, length(TrainingPoints)},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
	
	{set_testing_error, List} ->
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, [List|TestingError], MSPError);
	
	{reset_testing} ->
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, [], MSPError);
	
	{get_testing_error, From} ->
		From ! {ok, TestingError},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
		
	{set_mspe_error, ErrorValue} ->
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, [ErrorValue|MSPError]);
		
	{get_mspe_error, From} ->
		From ! {ok, MSPError},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, MSPError);
		
	{remove_mspe_error} ->
		NewMspe = hd(MSPError),
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError, [NewMspe])
		
end.

% Returns the Child Process Id of the selected Hidden Layer and Index
get_pid(Layer, Index) ->
	lists:nth(Index,Layer).
get_HL_pid(HLayerList, Index, InnerIndex) ->
	get_pid(lists:flatten(lists:nth(Index,HLayerList)),InnerIndex).