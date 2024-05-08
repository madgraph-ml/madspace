from madspace.diagram_mapping import *
from madspace.single_channel import SingleChannelVBS, SingleChannelWWW

MW = 80.377
WW = 2.085
MZ = 91.1876
WZ = 2.4952


def print_info(diagram):
    print("t vertices", diagram.t_channel_vertices)
    print("t lines   ", diagram.t_channel_lines)
    print("s vertices", diagram.s_channel_vertices)
    print("s lines   ", diagram.s_channel_lines)
    print("min sqrt s", [line.sqrt_s_min for line in diagram.s_channel_lines])


print("====== VBS ======")
in1 = Line()
in2 = Line()

out1 = Line(mass=MW)
out2 = Line()
out3 = Line()
out4 = Line(mass=MW)

t1 = Line()
t2 = Line()
t3 = Line()

v1 = Vertex([in1, out1, t3])
v2 = Vertex([t3, out2, t2])
v3 = Vertex([t2, out3, t1])
v4 = Vertex([t1, out4, in2])

r = torch.rand(3, 10)
vbsmap = SingleChannelVBS(torch.tensor(13000.**2), torch.tensor([MW]))
(p_hand, x_hand), jac_hand = vbsmap.map([r])

vbs = Diagram(incoming=[in1, in2], outgoing=[out1, out2, out3, out4], vertices=[v1, v2, v3, v4])
dmap = DiagramMapping(vbs, torch.tensor(13000.**2))
(p_auto, x_auto), jac_auto = dmap.map([r])

print_info(vbs)
print(p_auto - p_hand)
print(x_auto - x_hand)
print(jac_auto - jac_hand)

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

r = torch.rand(3, 7)
vbsmap = SingleChannelWWW(torch.tensor(13000.**2), torch.tensor(MW), torch.tensor(MZ), torch.tensor(WZ))
(p_hand, x_hand), jac_hand = vbsmap.map([r])

www = Diagram(incoming=[in1, in2], outgoing=[out1, out2, out3], vertices=[v1, v2, v3])
dmap = DiagramMapping(www, torch.tensor(13000.**2))
(p_auto, x_auto), jac_auto = dmap.map([r])

print(p_auto - p_hand)
print(x_auto - x_hand)
print(jac_auto - jac_hand)
print_info(www)
