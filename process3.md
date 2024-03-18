client -> server: input share
server -> client: model share

input is 3, bias is 5
client generates shares (5, 8) and sends 8 to server
server generates shares (1, 4) and sends 4 to client

client adds 5 (his shares) and 4 (received share) and gets 9
server adds 1 (his shares) and 8 (received share) and gets 9
server sends 9 (its result) to client
client adds 9 (its result) and 9 (received share) and gets 8
the result is 8.

-------------------------------------------------------------------------------

client is b = 0
server is b = 1

Input (x) is 3
Model is y = 2x + 4

Beaver's multiplication triple:
{u, v, uv} = (1, 9, 9)
client has {4, 2, 8}
server has {7, 7, 1}

=== INITIAL LAYER ===
client generates shares (5, 8) and sends 8 to server
server generates shares {(3, 9), (1, 3)} and sends {9, 3} to client

=== DENSE LAYER ===
client computes
<e> = 5 - 4
    = 1
<f> = 9 - 2
    = 7

server computes
<e> = 8 - 7
    = 1
<f> = 3 - 7
    = 6

client sends {1, 7} to server
server sends {1, 6} to client

client understands e = 2, f = 3
server understands e = 2, f = 3

client computes
<2x> = 3 * 4 + 2 * 2 + 8
     = 4
furthermore
<y> = 4 + 3
    = 7

server computes
<2x> = 2 * 3 + 3 * 7 + 2 * 7 + 1
     = 2
furthermore
<y> = 2 + 1
    = 3

=== FINAL LAYER ===

server sends 3 to client
client understands y = 0
