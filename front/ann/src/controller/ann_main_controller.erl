-module(ann_main_controller, [Req]).
-compile(export_all).

index('GET', []) ->
	Networks = boss_db:find(neuralnetworks, []),
	% A = list_to_integer("12343"),
	{ok, [{networks, Networks}]}.
	
create_network('GET', []) ->
	Name = Req:query_param("networkName"),
	Inputs = list_to_integer(Req:query_param("inputs")),
	Hidden = list_to_integer(Req:query_param("hidden")),
	Outputs = list_to_integer(Req:query_param("output")),
	
	NetworkPid = library:create_network(Inputs,[Hidden],Outputs),
	
	NewNet = neuralnetworks:new(id, Name, NetworkPid, Inputs, Hidden, Outputs),
	
	case NewNet:save() of
		{ok, SavedNet} ->
			{redirect, [{action, "index"}]}
	end.

pass_value('GET', []) ->
	Network = boss_db:find(Req:query_param("network_pid2")),
	NetworkPid = Network:network_pid(),
	
	library:pass_values(NetworkPid, [1,0.4,0.5], [0.4]),
	
	Output = library:get_output2(NetworkPid),
	
	NewOutput = lists:map(fun(Tup) -> {_,A,_,_} = Tup, A end, Output),
	
	{ok, [ {outputs, NewOutput} ]}.

pass_csv_training('GET', []) ->
	ok;
pass_csv_training('POST', []) ->
	Network = boss_db:find(Req:post_param("network_pid3")),
	NetworkPid = Network:network_pid(),
	
	% S = Req:post_files(),
	[{uploaded_file, FileName, Location, Length, _}] = Req:post_files(),
    Fname = "./priv/static/PSEData/" ++ FileName,
    file:copy(Location, Fname),
    file:delete(Location),
	
	% ExpectedValues = Req:post_param("trainValue"),
	
	% TokenizedValues = string:tokens(ExpectedValues, ", "),
	% Window = csvANN:stringToFloat(TokenizedValues, []),
	
	OutputLayer = length(library:get_output_layer(NetworkPid)),
	InputLayer = trunc(length(library:get_input_layer(NetworkPid))/4),
	
	io:format("~n~w ~w~n", [InputLayer, OutputLayer]),
	library:pass_training(Fname, NetworkPid, InputLayer, OutputLayer),
		
	{redirect, [{action, "index"}]}.
	
pass_csv_testing('GET', []) ->
	ok;
pass_csv_testing('POST', []) ->
	Network = boss_db:find(Req:post_param("network_pid5")),
	NetworkPid = Network:network_pid(),
	
	% S = Req:post_files(),
	% [{uploaded_file, FileName, Location, Length, _}] = Req:post_files(),
    % Fname = "./priv/static/PSEData/" ++ FileName,
    % file:copy(Location, Fname),
    % file:delete(Location),
	
	% ExpectedValues = Req:post_param("trainValue2"),
	
	% TokenizedValues = string:tokens(ExpectedValues, ", "),
	% Eval = csvANN:stringToFloat(TokenizedValues, []),
	
	library:pass_testing(NetworkPid),
		
	{redirect, [{action, "index"}]}.
	
pass_csv_forecast('GET', []) ->
	ok;
pass_csv_forecast('POST', []) ->
	Network = boss_db:find(Req:post_param("network_pid7")),
	NetworkPid = Network:network_pid(),
	NumOutputs = Network:num_outputs(),
	
	[{uploaded_file, FileName, Location, Length, _}] = Req:post_files(),
    Fname = "./priv/static/PSEData/" ++ FileName,
    file:copy(Location, Fname),
    file:delete(Location),
	
	% Eval = lists:map(fun(List) -> null end, lists:seq(1,NumOutputs)),
	LengthOutput = length(library:get_output_layer(NetworkPid)),
	
	library:pass_forecasting(Fname, NetworkPid, LengthOutput),
		
	{redirect, [{action, "index"}]}.
	
get_output_training('GET', []) ->
	Network = boss_db:find(Req:query_param("network_pid4")),
	NetworkPid = Network:network_pid(),
	OutputList = library:get_log_rev(NetworkPid),
	
	NewOutput = lists:map(fun(Tup) -> {_,Output,_,Actual} = Tup, {Output,Actual} end, OutputList),
	
	{ok, [{outputs, lists:reverse(NewOutput)}]}.
	
get_output_testing('GET', []) ->
	Network = boss_db:find(Req:query_param("network_pid6")),
	NetworkPid = Network:network_pid(),
	
	OutputList = library:mspe(NetworkPid),
	
	% NewOutput = lists:map(fun(Tup) -> {_,Output,Error,Actual} = Tup, {Output,Actual,Error} end, OutputList),
	
	{ok, [{outputs, OutputList}]}.
	
get_output_forecast('GET', []) ->
	Network = boss_db:find(Req:query_param("network_pid8")),
	NetworkPid = Network:network_pid(),
	
	OutputList = library:get_log_rev(NetworkPid),
	
	NewOutput = lists:map(fun(Tup) -> {_,Output,_,_} = Tup, Output end, OutputList),
	
	{ok, [{outputs, NewOutput}]}.
	
get_status('GET', []) ->
	Network = boss_db:find(Req:query_param("network_pid")),
	NetworkPid = Network:network_pid(),
	
	Status = library:get_status(NetworkPid, output, 1),

	{Pid, Weights, Inputs, Sensitivities, Stale_inputs, _} = Status,

	{ok,
		[
			{pid, io_lib:format("~w", [Pid])},
			{weights, io_lib:format("~w", [Weights])},
			{inputs, io_lib:format("~w", [Inputs])},
			{sensitivities, io_lib:format("~w", [Sensitivities])},
			{stale_inputs, io_lib:format("~w", [Stale_inputs])}
		]
	}.
	
