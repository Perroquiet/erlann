%%--------------------------------------------------------- 
%% @author MAGNUM TEAM
%% @copyright 2013 MAGNUM
%% @version 0.9.0 (lib)
%% @doc The Parent process function
%% @end
%%---------------------------------------------------------
-module(parent_ann).
-export([loop/7]).

%% @spec loop(InputLayer::list(), HiddenLayer::list(), OutputLayer::list(), Factors::list(), TrainingPoints::list(), TestingError::list(), MSPError::list()) -> none()
%% @doc Types:
%% ```
%% InputLayer = List of Input nodes
%% HiddenLayer = List of Hidden nodes
%% OutputLayer = List of Output nodes
%% Factors = List of tuples, containing the Denormalization Value, Expected Value, and the Log Expected Value
%% TrainingPoints = List of Training Points
%% TestingError = List of Testing Errors
%% MSPError = List of MSPE values
%% '''
%% Receive:
%% ```
%% {iLayer_Pid, Pid, Index} = returns a process id of the selected node in the Input layer
%% {hLayer_Pid, Pid, Index, InnerIndex} = returns a process id of the selected node in the Hidden layer
%% {oLayer_Pid, Pid, Index} = returns a process id of the selected node in the Output layer
%% {get_input_layer, Pid} = returns the list of Input nodes
%% {get_hidden_layer, Pid} = returns the list of Hidden nodes
%% {get_output_layer, Pid} = returns the list of Output nodes
%% {set_norm, DenormFactor, Exp, LogExp} = sets the Denormalization Factor, Expected Value, and Log Expected Value of the network
%% {get_norm, From} = returns  the  Factors tuple
%% {get_expected, From} = returns the expected values from Factors
%% {get_log_expected, From} = returns the log expected values from Factors
%% {return_status, NodePid, From} = returns the status of the perceptron
%% {return_output, NodePid, From} = returns the output of the preceptron
%% {pass, NodePid, InputValue, Expected_Val} = passes values to a perceptron
%% {set_training_points, Tp} = sets the Training Points of the Input Data
%% {get_training_point, From} = returns the list of Training Points
%% {count_training_set, From} = returns the number of Training Points
%% {set_testing_error, List} = sets the testing error during network testing
%% {reset_testing} = resets the testing error to an empty list
%% {get_testing_error, From} = returns a list of Testing Errors
%% {set_mspe_error, ErrorValue} = sets the MSPE error
%% {get_mspe_error, From} = returns the MSPE error
%% {remove_mspe_error} = removes the head of the list of MSPE errors
%% '''
%% The parent process of the neural network, that stores the state and configuration of the neural network.
%% @end

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