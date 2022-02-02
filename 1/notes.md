# Notes for assignment 1 

## Thoughts

Min max is altered because we want to max out score and minimize the opponent.

Terminal node: 
 - When available nodes == 0
 - When a player has won
 - Depth is == max_depth
 - When there is a draw

Can copy an entire environoment.

## Important vairables

in `play_game`, the variable `state` is the current board 

## Usefull things 

in `connect_four_env`: 
 - `env.board()` returns a copy of the current board, type np array
 - `env.available_moves()` returns a frozen set of available moves
 - 

## Hard part

In alpha beta, what is the variables:
 - node
 - depth
 - alpha
 - beta
 - maximizing_player

How to get child from node.

What is max depth

In student move, what are:
- origin 
- depth

How to evaluate different states of a board.

How to copy the current state to a new board.