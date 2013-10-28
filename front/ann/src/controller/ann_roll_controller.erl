-module(ann_roll_controller, [Req]).
-compile(export_all).

index('GET', []) ->
	{output, "hehe"}.