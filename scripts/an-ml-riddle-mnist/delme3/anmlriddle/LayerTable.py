# automatically generated by the FlatBuffers compiler, do not modify

# namespace: anmlriddle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class LayerTable(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsLayerTable(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LayerTable()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LayerTableBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x41\x4D\x52\x4D", size_prefixed=size_prefixed)

    # LayerTable
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LayerTable
    def LayerType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # LayerTable
    def Layer(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

def LayerTableStart(builder): builder.StartObject(2)
def LayerTableAddLayerType(builder, layerType): builder.PrependUint8Slot(0, layerType, 0)
def LayerTableAddLayer(builder, layer): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(layer), 0)
def LayerTableEnd(builder): return builder.EndObject()