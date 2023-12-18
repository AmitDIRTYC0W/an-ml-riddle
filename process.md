1. InitialLayer: Receive a model share from the server
2. InitialLayer: Send the input share to the server
3. DenseLayer: Sum the our input share with our layer share
4. FinalLayer: Server sends its final share to the client

Note: Maybe I should use a queue in case the server finished processing before the client
