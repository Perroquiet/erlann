-module(ann_graph_controller, [Req]).
-compile(export_all).
 
index('GET', []) ->
	{ok,[]}.
