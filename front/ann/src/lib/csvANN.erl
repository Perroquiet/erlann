-module(csvANN).

-export([
    parse/1, stringToFloat/2
]).

parse(File) ->
    try
        {ok, Bin} = file:read_file(File),
        {ok, parse(binary_to_list(Bin), [], [], [])}
    catch
        Class:Error ->
            {Class, Error}
    end.

parse([], _FBuff, _RBuff, Result) ->
    remove(lists:reverse(Result));
parse([$" | Rest], _FBuff, RBuff, Result) ->
    {F, Rest1} = parse_q(Rest, []),
    parse(Rest1, [], [F | RBuff], Result);
parse([$,, $\s| Rest], FBuff, RBuff, Result) ->
    parse(Rest, [], [lists:reverse(FBuff) | RBuff], Result);    
parse([$, | Rest], FBuff, RBuff, Result) ->
    parse(Rest, [], [lists:reverse(FBuff) | RBuff], Result);
parse([$\r, $\n | Rest], FBuff, RBuff, Result) ->
    parse(Rest, [], [], [lists:reverse([lists:reverse(FBuff)|RBuff]) | Result]);
parse([$\n | Rest], _FBuff, RBuff, Result) ->
    parse(Rest, [], [], [lists:reverse(RBuff) | Result]);
parse([A | Rest], FBuff, RBuff, Result) ->
    parse(Rest, [A | FBuff], RBuff, Result).

parse_q([$", $, | Rest], Result) ->
    {lists:reverse(Result), Rest};
parse_q([A | Rest], Result) ->
    parse_q(Rest, [A | Result]).
	
remove([_|T])->
removeHead(T, []).

removeHead([], A)->
stringToFloat(lists:append(A), []);
removeHead([H|T], A)->
removeHead(T,removeHeadAgain(H, A)).

removeHeadAgain([_|T], A)->
[T|A].

stringToFloat([], Acc)->
Acc;
stringToFloat([H|T], Acc)->
stringToFloat(T,[bin_to_num(H)|Acc]).

bin_to_num(Bin) ->
    case string:to_float(Bin) of
        {error,no_float} -> list_to_integer(Bin);
        {F,_Rest} -> F
    end.
