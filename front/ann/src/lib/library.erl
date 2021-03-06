% 
% Erlang ANN Library - Developer Version (v.0.9.0)
%
-module(library).
-compile(export_all).
-define(CSVCOLUMNS, 4).

%
% Parent Process Loop - This is where the state of the neural network is saved.
%
loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError) ->
receive
	{iLayer_Pid, Pid, Index} ->
		Pid ! {ok,get_PId(InputLayer,Index)},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
		
	{hLayer_Pid, Pid, Index, InnerIndex} ->	
		Pid ! {ok,get_HL_PId(HiddenLayer, Index, InnerIndex)},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
		
	{oLayer_Pid, Pid, Index} ->
		Pid ! {ok,get_PId(OutputLayer,Index)},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
		
	{get_input_layer, Pid} ->
		Pid ! {input_layer, InputLayer},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
	
	{get_hidden_layer, Pid} ->
		Pid ! {hidden_layer, HiddenLayer},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
	
	{get_output_layer, Pid} ->
		Pid ! {output_layer, OutputLayer},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
		
	{set_norm, DenormFactor, Exp, LogExp} ->
		loop(InputLayer, HiddenLayer, OutputLayer, [{DenormFactor,Exp,LogExp}], TrainingPoints, TestingError);
		
	{get_norm, From} ->
		From ! {ok, hd(Factors)},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
		
	{get_expected, From} ->
		P = hd(Factors),
		{_,ExpectedVal,_} = P,
		From ! {ok, ExpectedVal},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
		
	{get_log_expected, From} ->
		P = hd(Factors),
		{_,_,LogExpectedVal} = P,
		From ! {ok, LogExpectedVal},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
	
	{return_status, NodePid, From} ->
		NodePid ! {status, self()},
		receive
			{ok, Status} ->
				From ! {ok, Status}	
		end,
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
	
	{return_output, NodePid, From} ->
		NodePid ! {return_output, self()},
		receive
			{output, Pid, Output} ->
				From ! {ok, Pid, Output}
		end,
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
	
	{pass, NodePid, InputValue, Expected_Val} ->
		NodePid ! {pass, InputValue, Expected_Val},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);

	{set_training_points, Tp} ->
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, Tp, TestingError);
		
	{get_training_point, From} ->
		[H|T] = TrainingPoints,
		From ! {ok, H},
		io:format("Data Left: ~w~n", [length(TrainingPoints)]),
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, T, TestingError);
	
	{count_training_set, From} ->
		From ! {ok, length(TrainingPoints)},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError);
	
	{set_testing_error, List} ->
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, [List|TestingError]);
	
	{reset_testing} ->
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, []);
	
	{get_testing_error, From} ->
		From ! {ok, TestingError},
		loop(InputLayer, HiddenLayer, OutputLayer, Factors, TrainingPoints, TestingError)
end.


% MAJOR EXPORTED FUNCTIONS FOR THE NEURAL NETWORK CREATION AND EXECUTION


% Create network w/ number of inputs and hidden nodes.
% Returns a Network Pid - Parent Process Pid.
create_network(NumInputNodes,HiddenLayersList,NumOutputNodes) ->
	
	X = spawn_nodes(NumInputNodes,[]),
	Y = spawn_hidden(lists:map(fun(Nodes) -> Nodes+1 end, HiddenLayersList),[]),
	Z = spawn_nodes(NumOutputNodes,[]),
	io:format("Inputs:~w~nHidden:~w~nOutput:~w~n~n", [X,Y,Z]),
	connect_layer_to_layer(X,lists:nth(1,Y)),
	connect_hidden_layers(Y),
	connect_output_layers(lists:last(Y),Z),
	
	spawn(?MODULE,loop,[X,Y,Z,[],[],[]]).

% Terminate Parent Process of the network
destroy_network(NetworkPid) ->
	exit(NetworkPid,ok).

% Returns the Pid of the desired Node
get_pid(NetworkPid,Layer,Position) ->
	case Layer of
		input ->
			NetworkPid ! {iLayer_Pid,self(),Position},
			receive {ok,InputLPid} -> InputLPid end;
		hidden ->
			NetworkPid ! {hLayer_Pid,self(),lists:nth(1,Position),lists:last(Position)},
			receive {ok,HiddenLPid} -> HiddenLPid end;
		output ->
			NetworkPid ! {oLayer_Pid,self(),Position},
			receive {ok,OutputLPid} -> OutputLPid end
	end.
		

% Displays the status of the perceptron	
get_status(NetworkPid, Layer, Index) ->
	NetworkPid ! {return_status,get_pid(NetworkPid,Layer,Index),self()},
	receive
		{ok, Status} ->
			{Pid, Weights, Inputs, Sensitivities, Stale_inputs, Output} = Status,
			io:format("Status of Node(~w)~n Weights: ~w~n Inputs: ~w~n Sensitivities: ~w~n Stale Inputs: ~w~n Output: ~w~n", [Pid, Weights, Inputs, Sensitivities, Stale_inputs, Output])
			% Status
	end.

% Passes Values to the ANN with a List of Inputs, and List of Expected Values
pass_values(NetworkPid, Inputs, Expected) ->
	InputLayer = get_input_layer(NetworkPid),
	HiddenLayer = get_hidden_layer(NetworkPid),
	OutputLayer = get_output_layer(NetworkPid),
	backdone_buffer(length(InputLayer) * (length(hd(HiddenLayer))-1) * length(OutputLayer) + length(HiddenLayer), length(OutputLayer)),
	ExpVal =  lists:zip(OutputLayer,Expected),
	pass_hidden_biases(HiddenLayer, 1),
	set_norm(NetworkPid, [], Expected, Expected),
	pass_values(NetworkPid, InputLayer, Inputs, ExpVal).
pass_values(_, _, [], _) ->
	ok;
pass_values(_, [], _, _) ->
	ok;
pass_values(NetworkPid, [P|Q], [H|T], ExpectedVal) ->
	NetworkPid ! {pass, P, H, ExpectedVal},
	pass_values(NetworkPid, Q, T, ExpectedVal).
	
% Passes multiple values to a perceptron in a list fashion
pass_value_multi(_, _, [], _) ->
	ok;
pass_value_multi(_, [], _, _) ->
	ok;
pass_value_multi(_, ProcessList, DataList, ExpectedVal) ->	
	lists:foreach(fun({Data, Process}) ->
                          Process ! {pass, Data, ExpectedVal}
                  end,
                  lists:zip(DataList, ProcessList)).


% get_training_values(File) ->
	% {ok, Data} = csvANN:parse(File),
	% get_training_values(lists:nthtail(2 * 4, Data), 1, 4, 2, []).
	
get_training_values(_,_,_,0,Acc) ->
	lists:reverse(lists:flatten(Acc));
get_training_values(Data, Day, CSVColumns, TrainingDays, Acc) ->
	Eval = lists:sublist(Data, CSVColumns * Day, 1),
	get_training_values(Data, Day+1, CSVColumns, TrainingDays-1, [Eval|Acc]).
				  
get_training_points(Data, NetworkPid, TrainingWindowDays, TrainingDays) ->
	
	Window = TrainingWindowDays * ?CSVCOLUMNS,
	get_training_points(NetworkPid, Data, 0, ?CSVCOLUMNS, Window, TrainingDays, (length(Data)/4)+1 - (TrainingWindowDays + TrainingDays), []).
get_training_points(_,_,_,_,_,_,0.0,Acc) ->
	[X||{_,X} <- lists:sort([ {random:uniform(), N} || N <- Acc])];
	% file:write_file("dug",io_lib:fwrite("~p.\n", [lists:reverse(Acc)]));
get_training_points(NetworkPid, Data, HeadData, CSVColumns, Window, TrainingDays, Terminator, Acc) ->

	InputData = lists:sublist(Data, HeadData+1, Window),
	Eval = get_training_values(lists:nthtail(HeadData + Window,Data), 1, CSVColumns, TrainingDays, []),
	% io:format("~w~n", [Eval]),
	get_training_points(NetworkPid, Data, HeadData + CSVColumns, CSVColumns, Window, TrainingDays, Terminator-1, [{InputData, Eval}|Acc]).
	
	
pass_training(File, NetworkPid, TrainingWindowDays, TrainingDays) ->
	{ok, Data} = csvANN:parse(File),
	TrainingPoints = get_training_points(Data, NetworkPid, TrainingWindowDays, TrainingDays),
	NetworkPid ! {set_training_points, TrainingPoints},
	pass_training(NetworkPid, round(((length(Data)/4)+1 - (TrainingWindowDays + TrainingDays))* 0.8)).
pass_training(_,0) ->
	ok;
pass_training(NetworkPid, Terminator) ->
	
	InputLayer = get_input_layer(NetworkPid),
	HiddenLayer = get_hidden_layer(NetworkPid),
	OutputLayer = get_output_layer(NetworkPid),
	
	NetworkPid ! {get_training_point, self()},
	receive
		{ok, TrainingPoint} ->
			ok
	end,
	
	Pid = spawn(library, chorva, [NetworkPid, Terminator-1]),
	backdone_buffer(NetworkPid, length(InputLayer) * (length(hd(HiddenLayer))-1) * length(OutputLayer) + length(HiddenLayer), length(OutputLayer), Pid),
	{InputData,Eval} = TrainingPoint, 
	pass_recurrence(InputData, NetworkPid, Eval),
	
	io:format("Training Value: ~w~n~w~n", [Eval, Terminator]).
	
	
pass_testing(NetworkPid) ->
	
	NetworkPid ! {count_training_set, self()},
	receive
		{ok, Terminator} ->
			ok
	end,
	NetworkPid ! {reset_testing},
	pass_testing(NetworkPid, Terminator).
pass_testing(_,0) ->
	ok;
pass_testing(NetworkPid, Terminator) ->
	
	OutputLayer = get_output_layer(NetworkPid),
	
	NetworkPid ! {get_training_point, self()},
	receive
		{ok, TrainingPoint} ->
			ok
	end,
	{InputData,Eval} = TrainingPoint,
	
	% Test = lists:map(fun(_) -> null end, lists:seq(1,length(Eval))),
		
	Pid = spawn(library, chorva, [NetworkPid, Terminator-1]),
	backdone_buffer(NetworkPid, length(OutputLayer), length(OutputLayer), Pid),
	pass_recurrence2(InputData, NetworkPid, Eval).

pass_recurrence2(Data, NetworkPid, ExpectedVal) ->

		InputLayer = get_input_layer(NetworkPid),
		HiddenLayer = get_hidden_layer(NetworkPid),
		OutputLayer = get_output_layer(NetworkPid),	
		pass_hidden_biases(HiddenLayer, 1),
		
		
		Tails = length(Data) - 1,
		ConcatData = Data ++ ExpectedVal,
		Eval = lists:map(fun(_) -> null end, lists:seq(1,length(OutputLayer))),
		DenormFactors = lists:nthtail(Tails, remove(length(ConcatData),ConcatData)),
		NewData = log_norm(Data),
		set_norm(NetworkPid,DenormFactors, ExpectedVal, ExpectedVal),
		BiasData = [1|NewData],
		ExpVal =  lists:zip(OutputLayer,Eval),
		io:format("~w~n", [ExpVal]),
		training_buffer(ExpVal),
				% pass_value_multi(NetworkPid, InputLayer, BiasData, ExpVal);
		{T, _} = timer:tc(?MODULE, pass_value_multi, [NetworkPid, InputLayer, BiasData, ExpVal]),
		io:format("Time: ~w microseconds~n", [T]).
				
pass_forecasting(File, NetworkPid, WindowDays) ->
		OutputLayer = get_output_layer(NetworkPid),
		Window = WindowDays * ?CSVCOLUMNS,
		{ok,Data} = csvANN:parse(File),
		InputData = lists:nthtail(length(Data) - Window, Data),
		Eval = lists:map(fun(_) -> null end, lists:seq(1,length(OutputLayer))),
		pass_recurrence(InputData, NetworkPid, Eval).
				
chorva(NetworkPid, Terminator) ->
	receive
		{do_training} ->
			pass_training(NetworkPid, Terminator);
		{do_testing} ->
			pass_testing(NetworkPid, Terminator)
	end.
	
% Pass all values from a Philippine Stock Exchange (PSEi) formatted CSV Textfile
% This does not parse ordinary files
pass_recurrence(Data, NetworkPid, ExpectedVal) ->
% pass_from_textfile(File, NetworkPid, ExpectedVal) ->
		InputLayer = get_input_layer(NetworkPid),
		HiddenLayer = get_hidden_layer(NetworkPid),
		OutputLayer = get_output_layer(NetworkPid),
		% io:format("~w~n", [length(Data)]),
		% backdone_buffer(length(InputLayer) * (length(hd(HiddenLayer))-1) * length(OutputLayer) + length(HiddenLayer)),
		% {ok, Fd} = file:open(File, [read]),
		% Data = lists:map(fun(X) -> list_to_float(string:strip(X, right, $\n)) end, read(Fd)),
		% {ok, Data} = csvANN:parse(File),
		pass_hidden_biases(HiddenLayer, 1),
		
		case hd(ExpectedVal) of
			null ->
				NewData = log_norm(Data),
				set_norm(NetworkPid,lists:last(Data), ExpectedVal, ExpectedVal),
				BiasData = [1|NewData],
				ExpVal =  lists:zip(OutputLayer,ExpectedVal),
				% pass_value_multi(NetworkPid, InputLayer, BiasData, ExpVal);
				{T, _} = timer:tc(?MODULE, pass_value_multi, [NetworkPid, InputLayer, BiasData, ExpVal]),
				io:format("Time: ~w microseconds~n", [T]);
			_ ->
				Tails = length(Data) - 1,
				ConcatData = Data ++ ExpectedVal,
				NewData = log_norm(ConcatData),
				LogEVal = lists:nthtail(Tails, NewData),
				ExpVal =  lists:zip(OutputLayer,LogEVal),
				DenormFactors = lists:nthtail(Tails, remove(length(ConcatData),ConcatData)), 
				
				set_norm(NetworkPid,DenormFactors, ExpectedVal, LogEVal),
				BiasData = [1|NewData],
				InputData = lists:sublist(BiasData, 1,Tails+1),
				training_buffer(ExpVal),
				% io:format("~w~n", [ExpVal]),
				% pass_value_multi(NetworkPid, InputLayer, InputData, ExpVal)
				{T, _} = timer:tc(?MODULE, pass_value_multi, [NetworkPid, InputLayer, InputData, ExpVal]),
				io:format("Time: ~w microseconds~n", [T])
				
		end.

% Returns ta list of PIDs in the Input Layer		
get_input_layer(NetworkPid) ->
	NetworkPid ! {get_input_layer, self()},
	receive {input_layer, InputLayer} -> InputLayer end.

% Returns ta list of PIDs in the Hidden Layer(s)	
get_hidden_layer(NetworkPid) ->
	NetworkPid ! {get_hidden_layer, self()},
	receive {hidden_layer, HiddenLayer} -> HiddenLayer end.

% Returns ta list of PIDs in the Output Layer	
get_output_layer(NetworkPid) ->
	NetworkPid ! {get_output_layer, self()},
	receive {output_layer, OutputLayer} -> OutputLayer end.
	
% Gets the Weight configuration of the Network and saved to a textfile	
get_config(NetworkPid,FileName) ->
	X = get_input_layer(NetworkPid),
	Y = get_hidden_layer(NetworkPid),
	Z = get_output_layer(NetworkPid),

	A = get_config_layer(X),
	B = get_config_hidden(Y),
	C = get_config_layer(Z),
	Data = {A,B,C},
	file:write_file(FileName,io_lib:fwrite("~p.\n", [Data])),
	{A,B,C}.

	
% Loads the configuration from a textfile
load_config(ConfigFile) ->
	{ok,Content} = file:consult(ConfigFile),
	% string:len(Content),
	{A,B,C} = hd(Content),
	X = spawn_nodes(count_nodes(A),[]),
	Y = spawn_hidden(count_hidden_nodes(B),[]),
	Z = spawn_nodes(count_nodes(C),[]),
	io:format("Inputs:~w~nHidden:~w~nOutput:~w~n~n", [X,Y,Z]),
	connect_layer_to_layer(X,lists:nth(1,Y)),
	connect_hidden_layers(Y),
	connect_output_layers(lists:last(Y),Z),
	load_hidden_config(Y,B),
	load_layer_config(Z,C),
	spawn(?MODULE,loop,[X,Y,Z,[],[],[]]).

reset_weights(NetworkPid) ->
	Y = get_hidden_layer(NetworkPid),
	Z = get_output_layer(NetworkPid),
	
	lists:foreach(fun(Layer) -> lists:foreach(fun(Pid1) -> Pid1 ! {reset} end, Layer) end, Y),
	lists:foreach(fun(Pid2) -> Pid2 ! {reset} end, Z),
	ok.
	
% Returns a TupleList with each element 
% {Output_Process_ID, Output_from_the_network, Error}
get_output(NetworkPid) ->
	OutputLayer = get_output_layer(NetworkPid),
	LogEval = get_log_expected_val(NetworkPid),
	get_output(NetworkPid, OutputLayer, LogEval, []).
get_output(_, [], [], Outputs) ->
	lists:reverse(Outputs);
get_output(NetworkPid, [H|T], [P|Q], Outputs) ->
	NetworkPid ! {return_output, H, self()},
	receive
		{ok, Pid, Output} ->
			ok
	end,
	
	case P of
				null ->
					Error = null;
				_ ->
					
					Error = calculate_error(Output,P)
					% io:format("Output (~w): ~w with Error: ~w~n", [Pid, Output, Error])
	end,
	get_output(NetworkPid, T, Q, [{Pid,Output,Error,P}|Outputs]).

	
% Returns a tuplelist with each element is 
% {Output_node_Process_ID, Log_reversed_Output, Accuracy}
get_log_rev(NetworkPid) ->
	
	NetworkPid ! {get_norm, self()},
	receive
		{ok, Factors} ->
			{DenormFactor,Exp,_} = Factors
	end,
	get_log_rev(get_output(NetworkPid), DenormFactor, Exp, []).
get_log_rev([], _, _, OutputList) ->
	lists:reverse(OutputList);
get_log_rev([H|T], DenormFactor, [P|Q], Forecasts) when is_list(DenormFactor) == false ->
	{Pid,Output,_,_} = H,
	DenormalizedVal = log_reverse(Output,DenormFactor),
	io:format("Output (~w): ~w~n", [Pid, DenormalizedVal]),
	get_log_rev(T, DenormalizedVal, Q, [{Pid,DenormalizedVal,null,P}|Forecasts]);
get_log_rev([H|T], [P|Q], [R|S], OutputList) ->
	{Pid,Output,_,_} = H,
	DenormalizedVal = log_reverse(Output,P),
	case R of
			null ->
				Error = null,
				io:format("Output (~w): ~w~n", [Pid, DenormalizedVal]);
			X ->
				Error = calculate_error(DenormalizedVal, X),
				io:format("Output (~w): ~w with Accuracy: ~w~n", [Pid, DenormalizedVal, Error])
	end,
	get_log_rev(T, Q, S, [{Pid,DenormalizedVal,Error,R}|OutputList]).	

	
% Computes the Error of the network using the Error Fucntion:
% J(w) = 1/2 * SumOfAllOutputNodes((Training-Output)^2)
%
network_error(NetworkPid) ->
	Outputs = get_output(NetworkPid),
	Condition = hd(Outputs),
	{_,_,_,Eval} = Condition,
	% io:format("Eval: ~w~n", [Eval]),
	case Eval of
		null -> ok;
		_ ->
		Error = lists:map(fun({_,A,_,B}) -> math:pow((A-B),2) end, Outputs),
		% io:format("Error: ~w~n", [Error]),
		Total = lists:foldl(fun(X, Sum) -> X + Sum end, 0, Error),
		% io:format("Sum: ~w~n", [Total]),
		AveError = 0.5 * Total,
		
		io:format("Average Error for all output nodes: ~w~n", [AveError])
	end.
	

	

% Prints the Mean Squared Prediction Error
mspe(NetworkPid) ->
	% Outputs = get_log_rev(NetworkPid),
	% Condition = hd(Outputs),
	% {_,_,_,Eval} = Condition,
	
	NetworkPid ! {get_testing_error, self()},
	receive
		{ok, TestingError} ->
			% io:format("~w", [TestingError])
			ok
			
	end,
	% case Eval of
		% null ->
			% ok;
		% _ ->
			Outputs = lists:map(fun(Output) -> 
			Error = lists:map(fun({_,A,_,B}) -> math:pow((A-B),2) end, Output),
			lists:foldl(fun(X, Sum) -> X + Sum end, 0, Error)
			
			
			end, TestingError),
			Mspe = (lists:foldl(fun(X, Sum) -> X + Sum end, 0, Outputs))/length(Outputs),
			
			io:format("Mean Squared Prediction Error (MSPE): ~w~n", [Mspe]),
			Mspe.
	% end.


%	
% HELPER FUNCTIONS (NOT EXPORTED)
%

% CONFIGURATION

load_layer_config([],[]) ->
	ok;
load_layer_config([P|OLayer],[Q|OConfig]) ->
	P ! {load_config, get_weight_config(Q)},
	load_layer_config(OLayer,OConfig).
	
load_hidden_config([],[]) ->
	ok;
load_hidden_config([P|HLayer],[Q|HConfig]) ->
	load_hidden_weights(P,Q),
	load_hidden_config(HLayer,HConfig).

load_hidden_weights([],_) ->
	ok;
load_hidden_weights(_,[]) ->
	ok;
load_hidden_weights([H|T], [P|Q]) ->
	H ! {load_config, get_weight_config(P)},
	load_hidden_weights(T, Q).
	
get_weight_config(NodeConfig) ->
	{A} = NodeConfig,
	A.

count_hidden_nodes(HiddenLayers) ->
	count_hidden_nodes(HiddenLayers,[]).
count_hidden_nodes([], HiddenLayers) ->
	lists:reverse(HiddenLayers);
count_hidden_nodes([H|T], HiddenLayers) ->
	X = length(H),
	count_hidden_nodes(T, [X|HiddenLayers]).
	
count_nodes(Layer) ->
	length(Layer).

get_config_layer(List) ->
	get_config_layer(List, []).
get_config_layer([],List) ->
	lists:reverse(List);
get_config_layer([H|T], List) ->
	Config = ann:get_config(H),
	get_config_layer(T,[Config|List]).
	
get_config_hidden(HiddenLayers) ->
	get_config_hidden(HiddenLayers, []).
get_config_hidden([], HiddenLayersList) ->
	lists:reverse(HiddenLayersList);
get_config_hidden([H|T], HiddenLayersList) ->
	get_config_hidden(T, [get_config_layer(H)|HiddenLayersList]).
	
% NORMALIZATION

norm_max(List) ->
	norm_max(List, [], List).
norm_max([], Newlist, _) ->
	lists:reverse(Newlist);
norm_max([H|T], NewList, OrigList) ->
	X = H/lists:max(OrigList),
	norm_max(T, [X|NewList], OrigList).
	
denormalize(Output, MaxValue) -> 
	Output*MaxValue.
	
log_norm(List = [_H1,_H2|_T]) ->
    log_norm(List,[]);
log_norm(_) ->
    bad_argument.

log_norm([Input_prev,Input], Newlist) ->
    X = 100 * math:log10(Input/Input_prev), 
    lists:reverse([X|Newlist]);
log_norm([Input_prev,Input|T], Newlist) ->
    X = 100 * math:log10(Input/Input_prev), 
    log_norm([Input|T], [X|Newlist]).
	
log_reverse(Output, Val) ->
	X = math:pow(10, Output/100),
	X*Val.

calculate_error(Output, Expected) ->
	if
		Expected==0 ->
			Output;
		Output<Expected ->
			1 - (Output/Expected);
		Expected<Output ->
			1 - (Expected/Output);
		true ->
			0
	end.
	
% SPAWNING PERCEPTRONS
	
spawn_nodes(0,InputNodesList) ->
	lists:reverse(InputNodesList);
spawn_nodes(NumInputNodes,InputNodesList) when NumInputNodes>0 ->
	Pid = spawn(ann, perceptron, [[],[],[],[],[]]),
	spawn_nodes(NumInputNodes-1,[Pid|InputNodesList]).

	
spawn_hidden_bias(0,InputNodesList,_) ->
	lists:reverse(InputNodesList);
spawn_hidden_bias(NumInputNodes,InputNodesList,Bias) when Bias=:=true ->
	Pid = spawn(ann, perceptron, [[],[],[],[],[]]),
	spawn_hidden_bias(NumInputNodes-1,[Pid|InputNodesList],false);
spawn_hidden_bias(NumInputNodes,InputNodesList,Bias) when Bias=:=false ->
	Pid = spawn(ann, perceptron, [[],[],[],[],[]]),
	spawn_hidden_bias(NumInputNodes-1,[Pid|InputNodesList],false).	
	
spawn_hidden([], HiddenLayersList) ->
	lists:reverse(HiddenLayersList);
spawn_hidden([H|T], HiddenLayersList) ->
	spawn_hidden(T, [spawn_hidden_bias(H,[],true)|HiddenLayersList]).

% CONNECTING PERCEPTRONS	

connect_layer_to_layer([],_) ->
	ok;
connect_layer_to_layer(_,[]) ->
	ok;
connect_layer_to_layer([H|T], [P|Q]) ->
	connect_nodes(H,Q),
	connect_layer_to_layer(T,[P|Q]).

connect_output_layers([],_) ->
	ok;
connect_output_layers([H|T],List) ->
	connect_nodes(H,List),
	connect_output_layers(T,List).
	
connect_hidden_layers([]) ->
	ok;
connect_hidden_layers([H|T]) ->
	connect_layer_to_layer(H, lists:flatten(lists:sublist(T,1))),
	connect_hidden_layers(T).

connect_nodes(_, []) ->
	true;
connect_nodes(PID,[H|T]) ->
	ann:connect(PID, H),
	connect_nodes(PID, T).

read(File) ->
    case file:read_line(File) of
        {ok, Data} -> [Data | read(File)];
        eof        -> []
    end.

remove(N, L) -> remove(N, L, []). 
remove(_, [], Acc) -> lists:reverse(Acc); 
remove(1, [_|T], Acc) -> lists:reverse(Acc, T); 
remove(N, [H|T], Acc) -> remove(N-1, T, [H|Acc]). 
	

% Returns the Child Process Id of the selected Hidden Layer and Index
get_PId(Layer, Index) ->
	lists:nth(Index,Layer).
get_HL_PId(HLayerList, Index, InnerIndex) ->
	get_PId(lists:flatten(lists:nth(Index,HLayerList)),InnerIndex).

set_norm(NetworkPid, DenormFactor, ExpectedVal, LogEVal) ->
	NetworkPid ! {set_norm, DenormFactor, ExpectedVal, LogEVal}.
	
pass_hidden_biases([],_) ->
	ok;
pass_hidden_biases([H|T], Value) ->
	pass_value_node(hd(H), Value, 0),
	pass_hidden_biases(T, Value).
	
pass_value_node(Node, Value, ExpectedValue) ->
	Node ! {pass, Value, ExpectedValue}.
	

get_expected_val(NetworkPid) ->
	NetworkPid ! {get_expected, self()},
	receive
		{ok, Eval} ->
			Eval
	end.

get_log_expected_val(NetworkPid) ->
	NetworkPid ! {get_log_expected, self()},
	receive
		{ok, LogEval} ->
			LogEval
	end.
	
	
eval_buffer(Eval) ->
case lists:member(exp_val,registered()) of
		true ->
			unregister(exp_val);
		false ->
			ok
	end,
	register(exp_val, spawn(ann,expected_vals,[Eval])).

training_list(TrainList) ->
receive
	{get_list, From} ->
		From ! {ok, TrainList},
		training_list(TrainList)
end.

backdone_pid(Answers, Outputs) ->
receive
	{got_answer, From} ->
		case Answers of
			1 -> NewAnswers = 1,
				case Outputs of
					1 -> NewOutputs = 1, io:format("From: ~w Backpropagation Completed~n", [From]);
					_ -> NewOutputs = Outputs - 1
				end;
			_ -> NewAnswers = Answers - 1, NewOutputs = Outputs
		end,
		backdone_pid(NewAnswers, NewOutputs);
	{done_ff} ->
		case Answers of
			1 -> 
				NewAnswers = 1,
				io:format("Test Complete~n");
				% timer:sleep(50),
				% NetworkPid ! {set_testing_error, get_log_rev(NetworkPid)}, 
				% Pid ! {do_testing};
			_ -> NewAnswers = Answers - 1
		end,
		backdone_pid(NewAnswers, Outputs);
	{check_answers, From} ->
		case Answers of
			1 -> From ! {done};
			_ -> From ! {not_done}
		end,
		backdone_pid(Answers, Outputs)
end.
	
backdone_pid(NetworkPid, Answers, Outputs, Pid) ->
receive
	{got_answer, From} ->
		case Answers of
			1 -> NewAnswers = 1,
				case Outputs of
					1 -> NewOutputs = 1, io:format("From: ~w Backpropagation Completed~n", [From]), Pid ! {do_training};
					_ -> NewOutputs = Outputs - 1
				end;
			_ -> NewAnswers = Answers - 1, NewOutputs = Outputs
		end,
		backdone_pid(NetworkPid, NewAnswers, NewOutputs, Pid);
	{done_ff} ->
		case Answers of
			1 -> 
				NewAnswers = 1,
				io:format("Test Complete~n"),
				timer:sleep(50),
				NetworkPid ! {set_testing_error, get_log_rev(NetworkPid)}, 
				Pid ! {do_testing};
			_ -> NewAnswers = Answers - 1
		end,
		backdone_pid(NetworkPid, NewAnswers, Outputs, Pid);
	{check_answers, From} ->
		case Answers of
			1 -> From ! {done};
			_ -> From ! {not_done}
		end,
		backdone_pid(NetworkPid, Answers, Outputs, Pid)
		
end.
	
output_return(OutputList, DenormalizeFactor, ExpectedVal) ->
receive
	
	{output, Pid, Output} ->
		PidOutput = {Pid,Output},
		output_return([PidOutput|OutputList], DenormalizeFactor, ExpectedVal);
	{get_output} ->
		io:format("Output: ~w~n",[OutputList]);
	{log_reverse} ->
		lists:map(fun(Output) -> 
			{Pid, Value} = Output,
				log_rev_out(Pid, Value, DenormalizeFactor, ExpectedVal)
				end, 
			OutputList),
		output_return(OutputList, DenormalizeFactor, ExpectedVal)
		
end.

log_rev_out(Pid, Output, DenormalizeFactor, ExpectedVal) ->
	DenormalizedVal = log_reverse(Output,DenormalizeFactor),
		case ExpectedVal of
			null ->
				io:format("Output (~w): ~w~n", [Pid, DenormalizedVal]);
			X ->
				Error = calculate_error(DenormalizedVal, X),
				if
					Error=<0.1 ->
						io:format("Output (~w): ~w with Accuracy: ~w~n", [Pid, DenormalizedVal, Error]);
					true ->
						% text_data ! {pass_again} % LIMIT THE NUMBER OF PASSES
						io:format("Bent Output (~w): ~w with Accuracy: ~w~n", [Pid, DenormalizedVal, Error])
				end
		end.

backdone_buffer(Answers, Outputs) ->
	case lists:member(backdone_pid,registered()) of
		true ->
			unregister(backdone_pid);
		false ->
			ok
	end,
	register(backdone_pid, spawn(fun() -> backdone_pid(Answers, Outputs) end)).

backdone_buffer(NetworkPid, Answers, Outputs, Pid) ->
	case lists:member(backdone_pid,registered()) of
		true ->
			unregister(backdone_pid);
		false ->
			ok
	end,
	register(backdone_pid, spawn(fun() -> backdone_pid(NetworkPid, Answers, Outputs, Pid) end)).
	
training_buffer(TrainList) ->
	case lists:member(training_list,registered()) of
		true ->
			unregister(training_list);
		false ->
			ok
	end,
	register(training_list, spawn(fun() -> training_list(TrainList) end)).
	
		
output_buffer(Output,Expected) ->
	case lists:member(output_return,registered()) of
		true ->
			unregister(output_return);
		false ->
			ok
	end,
	register(output_return, spawn(fun() -> output_return([],Output,Expected) end)).

% Displays the output in denormalized form.
get_output2() ->
	output_return ! {get_output}.
	
get_stale_output(NetworkPid) ->
	OutputLayer = get_output_layer(NetworkPid),
	get_stale_output(NetworkPid, OutputLayer,[]).
get_stale_output(_, [], Outputs) ->
	lists:reverse(Outputs);
get_stale_output(NetworkPid, [H|T], Outputs) ->
	NetworkPid ! {return_output, H, self()},
	receive
		{ok, _, Output} ->
			ok
	end,
	get_stale_output(NetworkPid, T, [Output|Outputs]).
	
test_avg(M, F, A, N) when N > 0 ->
    L = test_loop(M, F, A, N, []),
    Length = length(L),
    Min = lists:min(L),
    Max = lists:max(L),
    Med = lists:nth(round((Length / 2)), lists:sort(L)),
    Avg = round(lists:foldl(fun(X, Sum) -> X + Sum end, 0, L) / Length),
    io:format("Range: ~b - ~b mics~n"
          "Median: ~b mics~n"
          "Average: ~b mics~n",
          [Min, Max, Med, Avg]),
    Med.
 
test_loop(_M, _F, _A, 0, List) ->
    List;
test_loop(M, F, A, N, List) ->
    {T, _Result} = timer:tc(M, F, A),
    test_loop(M, F, A, N - 1, [T|List]).
	