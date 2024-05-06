from madspace.diagram_mapping import *

MW = 80.377
WW = 2.085
MZ = 91.1876
WZ = 2.4952

print("====== VBS ======")
in1 = Line()
in2 = Line()

out1 = Line(mass=MW)
out2 = Line(mass=MW)
out3 = Line()
out4 = Line()

t1 = Line()
t2 = Line()
t3 = Line()

v1 = Vertex([in1, out1, t3])
v2 = Vertex([t3, out2, t2])
v3 = Vertex([t2, out3, t1])
v4 = Vertex([t1, out4, in2])

vbs = Diagram(incoming=[in1, in2], outgoing=[out1, out2, out3, out4], vertices=[v1, v2, v3, v4])

print("t vertices", vbs.t_channel_vertices)
print("t lines   ", vbs.t_channel_lines)
print("s vertices", vbs.s_channel_vertices)
print("s lines   ", vbs.s_channel_lines)

print()
print("====== WWW ======")

in1 = Line()
in2 = Line()

out1 = Line(mass=MW)
out2 = Line(mass=MW)
out3 = Line(mass=MW)

t1 = Line()
s1 = Line(mass=MZ, width=WZ)

v1 = Vertex([in1, s1, t1])
v2 = Vertex([t1, in2, out3])
v3 = Vertex([s1, out1, out2])

www = Diagram(incoming=[in1, in2], outgoing=[out1, out2, out3], vertices=[v1, v2, v3])

print("t vertices", www.t_channel_vertices)
print("t lines   ", www.t_channel_lines)
print("s vertices", www.s_channel_vertices)
print("s lines   ", www.s_channel_lines)
