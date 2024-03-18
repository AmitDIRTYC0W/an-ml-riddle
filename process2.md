client:

Inference:
	offlinePhase()
		Generate input shares
		Choose and process random numbers for Beaver's multiplication triples generation
	onlinePhase()
		DenseLayer
			After receiving the model share and enough MTs, calculate and store the result's share

DCF Key generation -> 
In DCF, the evaluation process involves
comparing a public input x ∈  Z_2^n to a private value a.

 F 1) Generate input shares
 F 2) Send the input shares
 F 3) Wait for the model share, the no. of MTs to generate, and the Beaver's multiplication triples prevalues 
SF 4) Generate enough MTs to calculate the first layer
 N 5) Calculate the first layer
SF 6) Generate more MTs until a new message is received or enough MTs were generated
....
