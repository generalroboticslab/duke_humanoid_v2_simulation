import numpy as np

elbow_long_info = """
General
	Part Number	motor mod
	Part Name	elbow-child_of_shoulder_3_joint_long v9
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.59956 kg
	Volume	0.00023 m^3
	Density	2576.46694 kg / m^3
	Area	0.15367 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.00313 m, -0.00028 m, -0.09729 m
	Bounding Box
		Length	0.0595 m
		Width	0.1036 m
		Height	0.1683 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.00091
		Ixy	-1.399E-08
		Ixz	2.851E-05
		Iyx	-1.399E-08
		Iyy	0.00081
		Iyz	-3.090E-06
		Izx	2.851E-05
		Izy	-3.090E-06
		Izz	0.00029
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.00659
		Ixy	5.125E-07
		Ixz	0.00021
		Iyx	5.125E-07
		Iyy	0.00649
		Iyz	-1.947E-05
		Izx	0.00021
		Izy	-1.947E-05
		Izz	0.0003
"""

wrist_1_long_info = """
General
	Part Number	motor mod
	Part Name	wrist_1-child_of_elbow_joint_long v8
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.689989903 kg
	Volume	0.000299827 m^3
	Density	2301.296826522 kg / m^3
	Area	0.183281567 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.000201373 m, -0.067392877 m, 0.023443516 m
	Bounding Box
		Length	0.111991825 m
		Width	0.1608 m
		Height	0.111991825 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.001083252
		Ixy	-6.268225219E-07
		Ixz	-8.869331290E-06
		Iyx	-6.268225219E-07
		Iyy	0.00062288
		Iyz	2.166055360E-05
		Izx	-8.869331290E-06
		Izy	2.166055360E-05
		Izz	0.000924216
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.004596265
		Ixy	-9.990733951E-06
		Ixz	-5.611969176E-06
		Iyx	-9.990733951E-06
		Iyy	0.001002126
		Iyz	0.001111794
		Izx	-5.611969176E-06
		Izy	0.001111794
		Izz	0.00405804
"""


shoulder_3_long_info = """
General
	Part Number	motor mod
	Part Name	shoulder_3-child_of_shoulder_2_joint_long v7
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.681053036 kg
	Volume	0.000308399 m^3
	Density	2208.353637942 kg / m^3
	Area	0.184925758 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.000450252 m, -0.067006035 m, 0.024534741 m
	Bounding Box
		Length	0.111991825 m
		Width	0.16277147 m
		Height	0.116492564 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.001085904
		Ixy	1.765262469E-06
		Ixz	-6.811209186E-06
		Iyx	1.765262469E-06
		Iyy	0.000597659
		Iyz	4.371421161E-05
		Izx	-6.811209186E-06
		Izy	4.371421161E-05
		Izz	0.000938729
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.004553664
		Ixy	2.231237207E-05
		Ixz	-1.433468084E-05
		Iyx	2.231237207E-05
		Iyy	0.001007759
		Iyz	0.001163349
		Izx	-1.433468084E-05
		Izy	0.001163349
		Izz	0.003996665
"""

wrist_2_long_info = """
General
	Part Number	motor mod
	Part Name	wrist_2-child_of_wrist_1_joint_long v8
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.403001309 kg
	Volume	0.000163222 m^3
	Density	2469.030993646 kg / m^3
	Area	0.100194773 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.005505358 m, -1.273393620E-05 m, -0.077506816 m
	Bounding Box
		Length	0.062008398 m
		Width	0.080610173 m
		Height	0.131305087 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.000357755
		Ixy	5.514576041E-08
		Ixz	-1.523595560E-05
		Iyx	5.514576041E-08
		Iyy	0.00034622
		Iyz	-1.004536913E-07
		Izx	-1.523595560E-05
		Izy	-1.004536913E-07
		Izz	0.000157663
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.002778707
		Ixy	2.689340105E-08
		Ixz	-0.000187198
		Iyx	2.689340105E-08
		Iyy	0.002779387
		Iyz	-4.982026253E-07
		Izx	-0.000187198
		Izy	-4.982026253E-07
		Izz	0.000169878
"""


############################SYMMETRIC FOOT #######################################
# ankle_2_info = """

# General
# 	Part Number	motor mod
# 	Part Name	ankle_2_symmetric v3
# 	Description
# 	Material Name	(Various)

# Manage
# 	Item Number
# 	Lifecycle
# 	Revision
# 	State
# 	Change Order

# Physical
# 	Mass	0.470005382 kg
# 	Volume	0.000316793 m^3
# 	Density	1483.637052247 kg / m^3
# 	Area	0.157622948 m^2
# 	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
# 	Center of Mass	-0.037567283 m, 0.00 m, 0.096157195 m
# 	Bounding Box
# 		Length	0.102008014 m
# 		Width	0.072014537 m
# 		Height	0.235007746 m
# 	Moment of Inertia at Center of Mass   (kg m^2)
# 		Ixx	0.003114704
# 		Ixy	0.00
# 		Ixz	7.838095907E-05
# 		Iyx	0.00
# 		Iyy	0.003300516
# 		Iyz	0.00
# 		Izx	7.838095907E-05
# 		Izy	0.00
# 		Izz	0.000570105
# 	Moment of Inertia at Origin   (kg m^2)
# 		Ixx	0.00746047
# 		Ixy	0.00
# 		Ixz	0.001776212
# 		Iyx	0.00
# 		Iyy	0.008309601
# 		Iyz	0.00
# 		Izx	0.001776212
# 		Izy	0.00
# 		Izz	0.001233423


# """

#################################################################################################

# ankle_2_info = """
# General
# 	Part Number	motor mod
# 	Part Name	ankle_2 v5
# 	Description
# 	Material Name	(Various)

# Manage
# 	Item Number
# 	Lifecycle
# 	Revision
# 	State
# 	Change Order

# Physical
# 	Mass	0.43189 kg
# 	Volume	0.00025 m^3
# 	Density	1729.35053 kg / m^3
# 	Area	0.17593 m^2
# 	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
# 	Center of Mass	-0.03975 m, 3.011E-05 m, 0.07119 m
# 	Bounding Box
# 		Length	0.0995 m
# 		Width	0.07201 m
# 		Height	0.1914 m
# 	Moment of Inertia at Center of Mass   (kg m^2)
# 		Ixx	0.00227
# 		Ixy	2.960E-07
# 		Ixz	8.366E-05
# 		Iyx	2.960E-07
# 		Iyy	0.00247
# 		Iyz	-6.216E-07
# 		Izx	8.366E-05
# 		Izy	-6.216E-07
# 		Izz	0.0005
# 	Moment of Inertia at Origin   (kg m^2)
# 		Ixx	0.00446
# 		Ixy	8.129E-07
# 		Ixz	0.00131
# 		Iyx	8.129E-07
# 		Iyy	0.00534
# 		Iyz	-1.547E-06
# 		Izx	0.00131
# 		Izy	-1.547E-06
# 		Izz	0.00118   
# """	

# symmetric foot
ankle_2_info = """
General
	Part Number	motor mod
	Part Name	ankle_2 v9
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.506070711 kg
	Volume	0.000338676 m^3
	Density	1494.264166543 kg / m^3
	Area	0.162911754 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.037477399 m, 8.960733679E-10 m, 0.095752247 m
	Bounding Box
		Length	0.102008962 m
		Width	0.07201851 m
		Height	0.237955922 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.003415165
		Ixy	0.00
		Ixz	0.000114866
		Iyx	0.00
		Iyy	0.003615949
		Iyz	0.00
		Izx	0.000114866
		Izy	0.00
		Izz	0.000615154
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.00805507
		Ixy	0.00
		Ixz	0.001930924
		Iyx	0.00
		Iyy	0.008966659
		Iyz	0.00
		Izx	0.001930924
		Izy	0.00
		Izz	0.001325958
"""


ankle_1_info = """
General
	Part Number	motor_mod
	Part Name	ankle_1 v11
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	1.7748727 kg
	Volume	0.000432293 m^3
	Density	4105.721153929 kg / m^3
	Area	0.315630485 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.030475314 m, -1.617012817E-05 m, -0.025612517 m
	Bounding Box
		Length	0.166 m
		Width	0.099140194 m
		Height	0.098160361 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.00142703
		Ixy	6.783650023E-07
		Ixz	-4.234522571E-06
		Iyx	6.783650023E-07
		Iyy	0.004181313
		Iyz	-5.458490150E-07
		Izx	-4.234522571E-06
		Izy	-5.458490150E-07
		Izz	0.004416667
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.002591349
		Ixy	-1.962740318E-07
		Ixz	-0.001389611
		Iyx	-1.962740318E-07
		Iyy	0.006994035
		Iyz	-1.280926182E-06
		Izx	-0.001389611
		Izy	-1.280926182E-06
		Izz	0.006065072
"""


shank_info = """
General
	Part Number	motor_mod
	Part Name	shank-child_of_knee_joint v6
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.725393968 kg
	Volume	0.000393933 m^3
	Density	1841.41282907 kg / m^3
	Area	0.22452484 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	8.920664668E-07 m, 0.094240522 m, 0.02074593 m
	Bounding Box
		Length	0.07138623 m
		Width	0.262947289 m
		Height	0.12101689 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.005775676
		Ixy	-9.155010957E-08
		Ixz	-2.982753971E-09
		Iyx	-9.155010957E-08
		Iyy	0.001447049
		Iyz	-1.140374753E-05
		Izx	-2.982753971E-09
		Izy	-1.140374753E-05
		Izz	0.004743825
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.012530305
		Ixy	-1.525331169E-07
		Ixz	-1.640743801E-08
		Iyx	-1.525331169E-07
		Iyy	0.001759254
		Iyz	-0.001429627
		Izx	-1.640743801E-08
		Izy	-0.001429627
		Izz	0.011186249
"""

knee_info = """
General
	Part Number	motor mod
	Part Name	knee-child_of_hip_3_joint v6
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	1.801868679 kg
	Volume	0.000495783 m^3
	Density	3634.392626063 kg / m^3
	Area	0.285546888 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.002711952 m, 8.606692555E-05 m, -0.073951458 m
	Bounding Box
		Length	0.085151025 m
		Width	0.16761627 m
		Height	0.174808158 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.003384709
		Ixy	-2.998805247E-06
		Ixz	2.484285332E-05
		Iyx	-2.998805247E-06
		Iyy	0.002474169
		Iyz	-4.009047396E-06
		Izx	2.484285332E-05
		Izy	-4.009047396E-06
		Izz	0.001771182
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.013238814
		Ixy	-3.419378287E-06
		Ixz	0.000386213
		Iyx	-3.419378287E-06
		Iyy	0.012341513
		Iyz	7.459440597E-06
		Izx	0.000386213
		Izy	7.459440597E-06
		Izz	0.001784448
"""

hip_3_info = """
General
	Part Number	motor-mod
	Part Name	hip_3-child of hip_2 joint v8
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	1.271048869 kg
	Volume	0.000404363 m^3
	Density	3143.338056051 kg / m^3
	Area	0.27601048 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.000300184 m, -0.077769961 m, 0.029074048 m
	Bounding Box
		Length	0.133375083 m
		Width	0.179278488 m
		Height	0.119719545 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.002470537
		Ixy	-3.216387844E-06
		Ixz	-7.843228438E-06
		Iyx	-3.216387844E-06
		Iyy	0.001619249
		Iyz	1.301282644E-05
		Izx	-7.843228438E-06
		Izy	1.301282644E-05
		Izz	0.002210427
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.011232471
		Ixy	2.645659867E-05
		Ixz	-1.893637834E-05
		Iyx	2.645659867E-05
		Iyy	0.002693782
		Iyz	0.002886966
		Izx	-1.893637834E-05
		Izy	0.002886966
		Izz	0.009898057
"""

hip_2_info = """
General
	Part Number	motor mod
	Part Name	hip_2-child of hip_1 joint v4
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	1.202700419 kg
	Volume	0.000357727 m^3
	Density	3362.061158323 kg / m^3
	Area	0.241268639 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.003943175 m, -2.277851562E-05 m, -0.061749796 m
	Bounding Box
		Length	0.079229473 m
		Width	0.137944958 m
		Height	0.147742877 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.001798658
		Ixy	-5.637729161E-07
		Ixz	3.254576296E-05
		Iyx	-5.637729161E-07
		Iyy	0.001460947
		Iyz	-2.084016348E-08
		Izx	3.254576296E-05
		Izy	-2.084016348E-08
		Izz	0.000973526
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.0063846
		Ixy	-4.557467569E-07
		Ixz	0.000325392
		Iyx	-4.557467569E-07
		Iyy	0.006065589
		Iyz	-1.712520915E-06
		Izx	0.000325392
		Izy	-1.712520915E-06
		Izz	0.000992227
"""

waist_info = """
General
	Part Number	motor_mod
	Part Name	waist v4
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	2.219371637 kg
	Volume	0.000735816 m^3
	Density	3016.202766966 kg / m^3
	Area	0.443342862 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.000128804 m, 3.828571812E-07 m, -0.071151903 m
	Bounding Box
		Length	0.130559359 m
		Width	0.159150314 m
		Height	0.1475615 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.005350555
		Ixy	-9.662510093E-07
		Ixz	8.287591919E-06
		Iyx	-9.662510093E-07
		Iyy	0.003250051
		Iyz	7.042220533E-08
		Izx	8.287591919E-06
		Izy	7.042220533E-08
		Izz	0.00478932
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.016586331
		Ixy	-9.661415640E-07
		Ixz	-1.205220616E-05
		Iyx	-9.661415640E-07
		Iyy	0.014485864
		Iyz	1.308801459E-07
		Izx	-1.205220616E-05
		Izy	1.308801459E-07
		Izz	0.004789357
"""

base_link_info = """
General
	Part Number	base_link
	Part Name	base_link v3
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	9.047510339 kg
	Volume	0.004943818 m^3
	Density	1830.065238427 kg / m^3
	Area	1.774840106 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.008491638 m, 5.800333285E-05 m, 0.185313814 m
	Bounding Box
		Length	0.143371336 m
		Width	0.1804 m
		Height	0.461146623 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.18448701
		Ixy	-3.131702671E-06
		Ixz	-0.003682535
		Iyx	-3.131702671E-06
		Iyy	0.169398462
		Iyz	1.701499112E-05
		Izx	-0.003682535
		Izy	1.701499112E-05
		Izz	0.034442935
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.495189489
		Ixy	1.324587893E-06
		Ixz	0.010554788
		Iyx	1.324587893E-06
		Iyy	0.480753308
		Iyz	-8.023505824E-05
		Izx	0.010554788
		Izy	-8.023505824E-05
		Izz	0.035095362
"""

wrist_3_info = """
General
	Part Number	motor mod
	Part Name	wrist_3-child_of_wrist_2_joint v4
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.31119 kg
	Volume	9.111E-05 m^3
	Density	3415.48369 kg / m^3
	Area	0.06956 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	7.018E-05 m, -0.04743 m, 0.0078 m
	Bounding Box
		Length	0.07504 m
		Width	0.10956 m
		Height	0.06011 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.00028
		Ixy	-1.065E-06
		Ixz	2.842E-07
		Iyx	-1.065E-06
		Iyy	0.00011
		Iyz	5.086E-05
		Izx	2.842E-07
		Izy	5.086E-05
		Izz	0.00027
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.001
		Ixy	-2.908E-08
		Ixz	1.138E-07
		Iyx	-2.908E-08
		Iyy	0.00013
		Iyz	0.00017
		Izx	1.138E-07
		Izy	0.00017
		Izz	0.00097
"""


shoulder_2_info = """
General
	Part Number	motor mod
	Part Name	shoulder_2-child_of_shoulder_1_joint v8
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.86409 kg
	Volume	0.00024 m^3
	Density	3591.0019 kg / m^3
	Area	0.16918 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.00285 m, -5.974E-07 m, -0.07394 m
	Bounding Box
		Length	0.0642 m
		Width	0.09503 m
		Height	0.14901 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.00148
		Ixy	-6.100E-09
		Ixz	3.807E-05
		Iyx	-6.100E-09
		Iyy	0.00133
		Iyz	-1.883E-08
		Izx	3.807E-05
		Izy	-1.883E-08
		Izz	0.00046
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.0062
		Ixy	-4.628E-09
		Ixz	0.00022
		Iyx	-4.628E-09
		Iyy	0.00606
		Iyz	-5.700E-08
		Izx	0.00022
		Izy	-5.700E-08
		Izz	0.00047
"""


end_effector_attachment_info = """
General
	Part Number	motor mod
	Part Name	end_effector_attachment v3
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.04111 kg
	Volume	1.346E-05 m^3
	Density	3053.50189 kg / m^3
	Area	0.01423 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.00497 m, -9.004E-09 m, 6.193E-10 m
	Bounding Box
		Length	0.026 m
		Width	0.04371 m
		Height	0.04371 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	8.902E-06
		Ixy	0.00
		Ixz	0.00
		Iyx	0.00
		Iyy	6.047E-06
		Iyz	0.00
		Izx	0.00
		Izy	0.00
		Izz	5.748E-06
	Moment of Inertia at Origin   (kg m^2)
		Ixx	8.902E-06
		Ixy	0.00
		Ixz	0.00
		Iyx	0.00
		Iyy	7.061E-06
		Iyz	0.00
		Izx	0.00
		Izy	0.00
		Izz	6.762E-06
"""

import re

def parse(info):
    get = lambda pat: float(re.search(pat, info).group(1))
    part_name = re.search(r'Part Name\s+(.+)', info).group(1).strip()
    part_number = re.search(r'Part Number\s+(.+)', info).group(1).strip()
    name = f"{part_name} ({part_number})"
    mass = get(r'Mass\s+([-\d.E]+)')
    com = [get(rf'Center of Mass\s+{p}') for p in [r'([-\d.E]+)', r'[-\d.E]+ m,\s+([-\d.E]+)', r'[-\d.E]+ m,\s+[-\d.E]+ m,\s+([-\d.E]+)']]
    # Extract inertia at center of mass (not at origin)
    com_section = re.search(r'Moment of Inertia at Center of Mass.*?(?=Moment of Inertia at Origin)', info, re.DOTALL).group(0)
    I = [float(re.search(rf'{k}\s+([-\d.E]+)', com_section).group(1)) for k in ['Ixx', 'Ixy', 'Ixz', 'Iyx', 'Iyy', 'Iyz', 'Izx', 'Izy', 'Izz']]
    print(f'link="{name}",')
    print(f'mass={mass},')
    print(f'inertia_origin={com},')
    # Align inertia columns
    cols = [[I[0], I[3], I[6]], [I[1], I[4], I[7]], [I[2], I[5], I[8]]]
    widths = [max(len(str(v)) for v in col) for col in cols]
    fmt = lambda r: ', '.join(f'{I[r*3+c]:>{widths[c]}}' for c in range(3))
    pad = ' ' * len('inertia=np.array([')
    print(f'inertia=np.array([[{fmt(0)}],')
    print(f'{pad}[{fmt(1)}],')
    print(f'{pad}[{fmt(2)}]]),')


# ─────────────────────────────────────────────────────────────────────
# Physics parser (merged from fusion_data_parser.py)
# ─────────────────────────────────────────────────────────────────────
from dataclasses import dataclass


@dataclass
class LinkPhysics:
    """Parsed Fusion360 physical properties for a link"""
    part_name: str
    mass: float       # kg
    com: np.ndarray   # [x, y, z] in metres
    inertia: np.ndarray  # 3×3 symmetric matrix at COM (kg·m²)

    def __repr__(self):
        return (f"LinkPhysics('{self.part_name}', "
                f"mass={self.mass:.5f} kg, "
                f"com=[{self.com[0]:.5f}, {self.com[1]:.5f}, {self.com[2]:.5f}])")


def parse_fusion_info(info_str: str) -> LinkPhysics:
    """Parse Fusion360 export string into LinkPhysics.

    Raises:
        ValueError: If required fields cannot be parsed
    """
    name_match = re.search(r'Part Name\s+(.+)', info_str)
    if not name_match:
        raise ValueError("Could not find 'Part Name' in fusion info")
    part_name = name_match.group(1).strip()

    mass_match = re.search(r'Mass\s+([\d.]+)\s*kg', info_str)
    if not mass_match:
        raise ValueError(f"Could not find Mass for {part_name}")
    mass = float(mass_match.group(1))

    com_match = re.search(
        r'Center of Mass\s+([-\d.Ee+]+)\s*m,\s*([-\d.Ee+]+)\s*m,\s*([-\d.Ee+]+)\s*m',
        info_str
    )
    if not com_match:
        raise ValueError(f"Could not find Center of Mass for {part_name}")
    com = np.array([float(com_match.group(i)) for i in range(1, 4)])

    inertia_section = re.search(
        r'Moment of Inertia at Center of Mass[^\n]*\n'
        r'\s*Ixx\s+([-\d.Ee+]+)\s*\n\s*Ixy\s+([-\d.Ee+]+)\s*\n\s*Ixz\s+([-\d.Ee+]+)\s*\n'
        r'\s*Iyx\s+([-\d.Ee+]+)\s*\n\s*Iyy\s+([-\d.Ee+]+)\s*\n\s*Iyz\s+([-\d.Ee+]+)\s*\n'
        r'\s*Izx\s+([-\d.Ee+]+)\s*\n\s*Izy\s+([-\d.Ee+]+)\s*\n\s*Izz\s+([-\d.Ee+]+)',
        info_str
    )
    if not inertia_section:
        raise ValueError(f"Could not find inertia tensor for {part_name}")

    ixx = float(inertia_section.group(1))
    ixy = float(inertia_section.group(2))
    ixz = float(inertia_section.group(3))
    iyy = float(inertia_section.group(5))
    iyz = float(inertia_section.group(6))
    izz = float(inertia_section.group(9))

    inertia = np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz]
    ])

    return LinkPhysics(part_name, mass, com, inertia)


def build_physics_database() -> dict:
    raw_data = {
        "ankle_1":          ankle_1_info,
        "ankle_2":          ankle_2_info,
        "base_link":        base_link_info,
        "end_effector_attachment": end_effector_attachment_info,
        "hip_2":            hip_2_info,
        "hip_3":            hip_3_info,
        "knee":             knee_info,
        "shank":            shank_info,
        "shoulder_2":       shoulder_2_info,
        "waist":            waist_info,
        "wrist_3":          wrist_3_info,
        # long versions
        "elbow_long":       elbow_long_info,
        "shoulder_3_long":  shoulder_3_long_info,
		"wrist_1_long":     wrist_1_long_info,
		"wrist_2_long":     wrist_2_long_info,
        
    }
    physics_db = {}
    for key, info_str in raw_data.items():
        try:
            physics_db[key] = parse_fusion_info(info_str)
        except ValueError as e:
            print(f"Warning: Failed to parse {key}: {e}")
    return physics_db


PHYSICS_DB = build_physics_database()


if __name__ == "__main__":
	print("=" * 60)
	print("Fusion360 Physics Database")
	print("=" * 60)

	for name, physics in sorted(PHYSICS_DB.items()):
		print(f"\n{name}:")
		print(f"  {physics}")
		print(f"  Inertia diagonal: [{physics.inertia[0,0]:.5f}, "
			  f"{physics.inertia[1,1]:.5f}, {physics.inertia[2,2]:.5f}]")

	print(f"\n{len(PHYSICS_DB)} parts successfully parsed")