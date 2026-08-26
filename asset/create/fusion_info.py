import numpy as np

elbow_long_info = """
General
	Part Number	motor mod
	Part Name	elbow-child_of_shoulder_3_joint_long v13
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.649565012 kg
	Volume	0.000234566 m^3
	Density	2769.218061792 kg / m^3
	Area	0.155616163 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.001650159 m, 0.002444203 m, -0.094370856 m
	Bounding Box
		Length	0.063627904 m
		Width	0.119192536 m
		Height	0.168300213 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.001065707
		Ixy	2.538250400E-05
		Ixz	6.314092617E-05
		Iyx	2.538250400E-05
		Iyy	0.000921032
		Iyz	-5.680591399E-05
		Izx	6.314092617E-05
		Izy	-5.680591399E-05
		Izz	0.00038905
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.006854522
		Ixy	2.276259858E-05
		Ixz	0.000164296
		Iyx	2.276259858E-05
		Iyy	0.006707735
		Iyz	9.302374321E-05
		Izx	0.000164296
		Izy	9.302374321E-05
		Izz	0.0003947
"""

wrist_1_long_info = """
General
	Part Number	motor mod
	Part Name	wrist_1-child_of_elbow_joint_long v21
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.745373662 kg
	Volume	0.000306554 m^3
	Density	2431.460063863 kg / m^3
	Area	0.187119702 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.000186448 m, -0.065153303 m, 0.003016967 m
	Bounding Box
		Length	0.111991825 m
		Width	0.1688 m
		Height	0.113482685 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.001297968
		Ixy	-9.364471707E-07
		Ixz	-8.813620802E-06
		Iyx	-9.364471707E-07
		Iyy	0.000709757
		Iyz	-4.417903871E-05
		Izx	-8.813620802E-06
		Izy	-4.417903871E-05
		Izz	0.001051551
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.004468828
		Ixy	-9.991038307E-06
		Ixz	-8.394341942E-06
		Iyx	-9.991038307E-06
		Iyy	0.000716568
		Iyz	0.000102336
		Izx	-8.394341942E-06
		Izy	0.000102336
		Izz	0.004215653
"""


shoulder_3_long_info = """
General
	Part Number	motor mod
	Part Name	shoulder_3-child_of_shoulder_2_joint_long v11
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.716054399 kg
	Volume	0.000310095 m^3
	Density	2309.145265696 kg / m^3
	Area	0.187215979 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.000428244 m, -0.065720173 m, 0.026903765 m
	Bounding Box
		Length	0.111991825 m
		Width	0.16277147 m
		Height	0.116492564 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.001229158
		Ixy	2.159566441E-06
		Ixz	-6.084758218E-06
		Iyx	2.159566441E-06
		Iyy	0.000675979
		Iyz	1.271237578E-06
		Izx	-6.084758218E-06
		Izy	1.271237578E-06
		Izz	0.001003874
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.004840187
		Ixy	2.231237207E-05
		Ixz	-1.433468084E-05
		Iyx	2.231237207E-05
		Iyy	0.001194399
		Iyz	0.001267341
		Izx	-1.433468084E-05
		Izy	0.001267341
		Izz	0.004096745
"""

wrist_2_long_info = """
General
	Part Number	motor mod
	Part Name	good_wrist_2-child_of_wrist_1_joint_long v2
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.446371412 kg
	Volume	0.000182453 m^3
	Density	2446.503074042 kg / m^3
	Area	0.110668818 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.00845747 m, -2.178259401E-05 m, -0.078162808 m
	Bounding Box
		Length	0.067008398 m
		Width	0.08108085 m
		Height	0.135805087 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.000379599
		Ixy	3.080214615E-08
		Ixz	-1.707723143E-05
		Iyx	3.080214615E-08
		Iyy	0.000403138
		Iyz	-1.939427842E-07
		Izx	-1.707723143E-05
		Izy	-1.939427842E-07
		Izz	0.000204436
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.003106672
		Ixy	-5.143090659E-08
		Ixz	-0.000312155
		Iyx	-5.143090659E-08
		Iyy	0.003162139
		Iyz	-9.539297155E-07
		Izx	-0.000312155
		Izy	-9.539297155E-07
		Izz	0.000236364
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
	Part Name	ankle_1 v16
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	1.801864964 kg
	Volume	0.000433574 m^3
	Density	4155.843829891 kg / m^3
	Area	0.316799575 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.030588768 m, -1.592789672E-05 m, -0.025773968 m
	Bounding Box
		Length	0.166 m
		Width	0.099140194 m
		Height	0.098160361 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.001445426
		Ixy	6.816211461E-07
		Ixz	2.333852097E-06
		Iyx	6.816211461E-07
		Iyy	0.004213609
		Iyz	-5.412153978E-07
		Izx	2.333852097E-06
		Izy	-5.412153978E-07
		Izz	0.004438067
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.0026424
		Ixy	-1.962740318E-07
		Ixz	-0.001418246
		Iyx	-1.962740318E-07
		Iyy	0.00709654
		Iyz	-1.280926182E-06
		Izx	-0.001418246
		Izy	-1.280926182E-06
		Izz	0.006124023
"""


shank_info = """
General
	Part Number	motor_mod
	Part Name	shank-child_of_knee_joint v8
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.753288544 kg
	Volume	0.000394771 m^3
	Density	1908.163761454 kg / m^3
	Area	0.225356775 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	8.590328886E-07 m, 0.094375148 m, 0.022990586 m
	Bounding Box
		Length	0.07138623 m
		Width	0.262947289 m
		Height	0.12101689 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.005991173
		Ixy	-9.146299298E-08
		Ixz	-1.530238200E-09
		Iyx	-9.146299298E-08
		Iyy	0.001545724
		Iyz	-1.733155681E-05
		Izx	-1.530238200E-09
		Izy	-1.733155681E-05
		Izz	0.004860768
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.013098628
		Ixy	-1.525331169E-07
		Ixz	-1.640743801E-08
		Iyx	-1.525331169E-07
		Iyy	0.001943888
		Iyz	-0.001651772
		Izx	-1.640743801E-08
		Izy	-0.001651772
		Izz	0.011570059
"""

knee_info = """
General
	Part Number	motor mod
	Part Name	knee-child_of_hip_3_joint v8
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	1.842141671 kg
	Volume	0.000497378 m^3
	Density	3703.707528606 kg / m^3
	Area	0.287701993 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.001482603 m, 8.418532619E-05 m, -0.073660261 m
	Bounding Box
		Length	0.114 m
		Width	0.16761627 m
		Height	0.174808158 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.003417182
		Ixy	-3.189454334E-06
		Ixz	6.437991687E-05
		Iyx	-3.189454334E-06
		Iyy	0.002647016
		Iyz	-3.963888192E-06
		Izx	6.437991687E-05
		Izy	-3.963888192E-06
		Izz	0.001911777
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.01341235
		Ixy	-3.419378287E-06
		Ixz	0.000265558
		Iyx	-3.419378287E-06
		Iyy	0.01264622
		Iyz	7.459440597E-06
		Izx	0.000265558
		Izy	7.459440597E-06
		Izz	0.001915839
"""

hip_3_info = """
General
	Part Number	motor mod
	Part Name	hip_3-child of hip_2 joint v11
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	1.297335685 kg
	Volume	0.000404552 m^3
	Density	3206.846411867 kg / m^3
	Area	0.276166679 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	2.077309868E-05 m, -0.077061422 m, 0.030723147 m
	Bounding Box
		Length	0.133375083 m
		Width	0.18103475 m
		Height	0.134692371 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.002715645
		Ixy	-1.732444781E-07
		Ixz	2.516026668E-07
		Iyx	-1.732444781E-07
		Iyy	0.001754008
		Iyz	-4.240746914E-05
		Izx	2.516026668E-07
		Izy	-4.240746914E-05
		Izz	0.002291367
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.011644396
		Ixy	1.903536367E-06
		Ixz	-5.763763864E-07
		Iyx	1.903536367E-06
		Iyy	0.002978579
		Iyz	0.003029125
		Izx	-5.763763864E-07
		Izy	0.003029125
		Izz	0.009995547
"""

hip_2_info = """
General
	Part Number	motor mod
	Part Name	hip_2-child of hip_1 joint v8
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	1.252698981 kg
	Volume	0.000361707 m^3
	Density	3463.301236903 kg / m^3
	Area	0.246602927 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.000349087 m, -2.186936422E-05 m, -0.061260977 m
	Bounding Box
		Length	0.166814736 m
		Width	0.137944958 m
		Height	0.149021034 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.001845915
		Ixy	-4.653102490E-07
		Ixz	7.531332837E-05
		Iyx	-4.653102490E-07
		Iyy	0.001928876
		Iyz	-3.423172298E-08
		Izx	7.531332837E-05
		Izy	-3.423172298E-08
		Izz	0.00139448
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.006547178
		Ixy	-4.557467569E-07
		Ixz	0.000102103
		Iyx	-4.557467569E-07
		Iyy	0.006630291
		Iyz	-1.712520915E-06
		Izx	0.000102103
		Izy	-1.712520915E-06
		Izz	0.001394633
"""

waist_info = """
General
	Part Number	motor_mod
	Part Name	waist v6
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	2.319367147 kg
	Volume	0.000746168 m^3
	Density	3108.371399381 kg / m^3
	Area	0.457233982 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.002382069 m, 4.634745432E-07 m, -0.070472052 m
	Bounding Box
		Length	0.142532017 m
		Width	0.159150314 m
		Height	0.1475615 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.005563804
		Ixy	-9.537623817E-07
		Ixz	6.737990144E-05
		Iyx	-9.537623817E-07
		Iyy	0.003762715
		Iyz	6.706492969E-08
		Izx	6.737990144E-05
		Izy	6.706492969E-08
		Izz	0.005315983
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.017082501
		Ixy	-9.512017345E-07
		Ixz	-0.000321971
		Iyx	-9.512017345E-07
		Iyy	0.015294572
		Iyz	1.428201042E-07
		Izx	-0.000321971
		Izy	1.428201042E-07
		Izz	0.005329144
"""

base_link_info = """
General
	Part Number	motor mod
	Part Name	base_link v5
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State	Working
	Change Order

Physical
	Mass	9.302908449 kg
	Volume	0.004943818 m^3
	Density	1881.725328946 kg / m^3
	Area	1.774840106 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	-0.007416363 m, 4.161614119E-05 m, 0.186842952 m
	Bounding Box
		Length	0.143371336 m
		Width	0.1804 m
		Height	0.461146623 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.18890243
		Ixy	-7.140666184E-06
		Ixz	-0.00480249
		Iyx	-7.140666184E-06
		Iyy	0.173948741
		Iyz	4.926105611E-05
		Izx	-0.00480249
		Izy	4.926105611E-05
		Izz	0.034929056
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.513669666
		Ixy	-4.269412568E-06
		Ixz	0.008088506
		Iyx	-4.269412568E-06
		Iyy	0.499227644
		Iyz	-2.307540796E-05
		Izx	0.008088506
		Izy	-2.307540796E-05
		Izz	0.035440754
"""

# wrist_3_info = """
# General
# 	Part Number	motor mod
# 	Part Name	wrist_3-child_of_wrist_2_joint v4
# 	Description
# 	Material Name	(Various)

# Manage
# 	Item Number
# 	Lifecycle
# 	Revision
# 	State
# 	Change Order

# Physical
# 	Mass	0.31119 kg
# 	Volume	9.111E-05 m^3
# 	Density	3415.48369 kg / m^3
# 	Area	0.06956 m^2
# 	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
# 	Center of Mass	7.018E-05 m, -0.04743 m, 0.0078 m
# 	Bounding Box
# 		Length	0.07504 m
# 		Width	0.10956 m
# 		Height	0.06011 m
# 	Moment of Inertia at Center of Mass   (kg m^2)
# 		Ixx	0.00028
# 		Ixy	-1.065E-06
# 		Ixz	2.842E-07
# 		Iyx	-1.065E-06
# 		Iyy	0.00011
# 		Iyz	5.086E-05
# 		Izx	2.842E-07
# 		Izy	5.086E-05
# 		Izz	0.00027
# 	Moment of Inertia at Origin   (kg m^2)
# 		Ixx	0.001
# 		Ixy	-2.908E-08
# 		Ixz	1.138E-07
# 		Iyx	-2.908E-08
# 		Iyy	0.00013
# 		Iyz	0.00017
# 		Izx	1.138E-07
# 		Izy	0.00017
# 		Izz	0.00097
# """


shoulder_2_info = """
General
	Part Number	motor mod
	Part Name	shoulder_2-child_of_shoulder_1_joint v13
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.903096525 kg
	Volume	0.000245765 m^3
	Density	3674.637430837 kg / m^3
	Area	0.174991325 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.000787068 m, -5.714548924E-07 m, -0.073008145 m
	Bounding Box
		Length	0.083601597 m
		Width	0.095025216 m
		Height	0.149009067 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.001526196
		Ixy	-5.028213923E-09
		Ixz	7.974803397E-05
		Iyx	-5.028213923E-09
		Iyy	0.001466961
		Iyz	-1.931632383E-08
		Izx	7.974803397E-05
		Izy	-1.931632383E-08
		Izz	0.000562177
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.006339872
		Ixy	-4.622024457E-09
		Ixz	0.000131642
		Iyx	-4.622024457E-09
		Iyy	0.006281196
		Iyz	-5.699428899E-08
		Izx	0.000131642
		Izy	-5.699428899E-08
		Izz	0.000562737
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


wrist_3_new_info ="""
General
	Part Number	motor_mod
	Part Name	new-wrist_3-child_of_wrist_2_joint v3
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.352354158 kg
	Volume	0.000200368 m^3
	Density	1758.538738461 kg / m^3
	Area	0.085120828 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.006987312 m, -0.009075624 m, -0.047417055 m
	Bounding Box
		Length	0.071904943 m
		Width	0.071818184 m
		Height	0.131048242 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	0.000405012
		Ixy	-1.866091238E-05
		Ixz	-0.00010307
		Iyx	-1.866091238E-05
		Iyy	0.000432328
		Iyz	-4.811652582E-05
		Izx	-0.00010307
		Izy	-4.811652582E-05
		Izz	0.000173439
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.001226259
		Ixy	3.683352600E-06
		Ixz	1.367109331E-05
		Iyx	3.683352600E-06
		Iyy	0.001241755
		Iyz	-0.000199748
		Izx	1.367109331E-05
		Izy	-0.000199748
		Izz	0.000219665
"""

end_effector_attachment_new_info = """
General
	Part Number	motor mod
	Part Name	new-end_effector_attachment v15
	Description
	Material Name	(Various)

Manage
	Item Number
	Lifecycle
	Revision
	State
	Change Order

Physical
	Mass	0.17299 kg
	Volume	0.00014 m^3
	Density	1262.18476 kg / m^3
	Area	0.04011 m^2
	World X,Y,Z	0.00 m, 0.00 m, 0.00 m
	Center of Mass	0.01513 m, -2.418E-08 m, -0.01529 m
	Bounding Box
		Length	0.098 m
		Width	0.066 m
		Height	0.05551 m
	Moment of Inertia at Center of Mass   (kg m^2)
		Ixx	6.366E-05
		Ixy	1.250E-10
		Ixz	-2.040E-05
		Iyx	1.250E-10
		Iyy	0.00014
		Iyz	0.00
		Izx	-2.040E-05
		Izy	0.00
		Izz	0.00014
	Moment of Inertia at Origin   (kg m^2)
		Ixx	0.0001
		Ixy	1.883E-10
		Ixz	1.961E-05
		Iyx	1.883E-10
		Iyy	0.00022
		Iyz	0.00
		Izx	1.961E-05
		Izy	0.00
		Izz	0.00018
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
        "end_effector_attachment_new": end_effector_attachment_new_info,
        "wrist_3_new":       wrist_3_new_info,
        
        "hip_2":            hip_2_info,
        "hip_3":            hip_3_info,
        "knee":             knee_info,
        "shank":            shank_info,
        "shoulder_2":       shoulder_2_info,
        "waist":            waist_info,
        # "wrist_3":          wrist_3_info,
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