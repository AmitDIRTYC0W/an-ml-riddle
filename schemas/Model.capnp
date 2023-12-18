@0xf37748307a09b723;

using Com = UInt16;

struct VectorShare {
  vectorShare @0 :List(Com);
}

struct ModelShare {
  layers @0 :List(LayerShare);
}

struct LayerShare {
  struct Dense {
    weightsShare @0 :List(Com);
    biasesShare @1 :List(Com);
  }
  
  union {
    dense @0 :Dense;
    fuck @1 :Void;
  }
}
