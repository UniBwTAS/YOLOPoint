uint64                       object_id                  # unique identifier for the object or track identifier.
string                       description                # JSON-string with different members, e.g. 'name', 'license_plate', 'color', ...
Pose64                       pose                       # 6 DOF pose, for object's reference point (REF).
Pose32                       velocity                   # [m/s, rad/s] 6d object velocity, given in object's coordinate system w.r.t. ground.
Pose32                       acceleration               # [m/s^2, rad/s^2] 6d object acceleration, given in object's coordinate system.
ObjectDimension              dimension                  # object's bounding-box dimensions from REF.
bool[24]                     state_validity             # Only elements which are true have a valid state value. The covariance matrix will consist of this entries only! For index values, see ObjectStateIndex.msg.
float32[]                    complete_covariance        # dynamic-size covariance matrix, given as upper triangular matrix and reduced to valid states only. Stored in row-major order.
uint32                       reference_point_type       # type of reference point to which 'pose' refers. For definition see ReferencePointType.msg!
MotionType                   motion_type                # For definition see MotionType.msg!
uint32                       object_class               # the most certain class, mandatory, e.g. UNKNOWN. For definition see ObjectClass.msg!
ObjectClassEstimate[]        object_class_estimate      # histogram of potential classes (optional).
float32                      existence_probability      # value between 0 and 1; -1 if not set.
time                         timestamp                  # [s] Timestamp of last measurement update, may differ between objects in same list. 0 if not set.
