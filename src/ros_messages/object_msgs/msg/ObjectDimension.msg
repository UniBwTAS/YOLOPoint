# All values are defined w.r.t. the reference point of the object, e.g. the pose field in Object.msg. This allows to
# use different reference points depending on the current aspect angle to avoid ghost motion due to changes in the
# object dimensions from time step t to t+1.

float32 length_to_front      # [m] normally positive if REF is inside object
float32 length_to_rear       # [m] normally negative if REF is inside object
float32 width_to_left        # [m] normally positive if REF is inside object
float32 width_to_right       # [m] normally negative if REF is inside object
float32 height_to_upper      # [m] normally positive if REF is inside object
float32 height_to_bottom     # [m] normally negative if REF is inside object
