# automatically generated by the FlatBuffers compiler, do not modify

# namespace: anmlriddle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Dense(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsDense(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Dense()
        x.Init(buf, n + offset)
        return x

    # Dense
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Dense
    def Values(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # Dense
    def ValuesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int16Flags, o)
        return 0

    # Dense
    def ValuesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Dense
    def ValuesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def DenseStart(builder): builder.StartObject(1)
def DenseAddValues(builder, values): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(values), 0)
def DenseStartValuesVector(builder, numElems): return builder.StartVector(2, numElems, 2)
def DenseEnd(builder): return builder.EndObject()
