# Don't include in other msgs, as this msg is subject to regular changes, e.g. when new classes are added. This is
# important because changing messages causes problems in existing bag files. So the idea is that the object_class field
# in the Object.msg or the class field in ObjectClassEstimate.msg is set to one of the constants defined here in this
# message with a simple int32 but does not include this message directly. This message exists to list all available
# classes and can be used in the source code.

uint32 UNKNOWN               = 0
uint32 CAR                   = 1
uint32 TRUCK                 = 2
uint32 BICYCLE               = 3
uint32 MOTORCYCLE            = 4
uint32 PEDESTRIAN            = 5
uint32 TREE                  = 6
uint32 BUSH                  = 7
uint32 TRAFFIC_SIGN          = 8
uint32 ERICARD               = 9
uint32 BUS                   = 10
uint32 TRAILER               = 11
uint32 CONSTRUCTION_VEHICLE  = 12
uint32 TRAFFIC_CONE          = 13
uint32 BARRIER               = 14

uint32 val
