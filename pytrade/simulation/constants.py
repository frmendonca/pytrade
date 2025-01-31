from scipy.stats import norm


NUMERIC_INFINITY = 9e10

# Tail generator parameters
LEFT_TAIL_S = 0.9274826457963914
LEFT_TAIL_LOC = -0.003154933610007448
LEFT_TAIL_SCALE = 0.027694808964929315

RIGHT_TAIL_S = 0.5853417322532056
RIGHT_TAIL_LOC = -0.009161941839938244
RIGHT_TAIL_SCALE = 0.035561782229894215

MIXTURE_PARAMETER = 0.6376351467307362

# VIX/SPY Relation
SPY_BINS =[-1.0, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10,NUMERIC_INFINITY]
VIX_GENERATORS = [
    norm(loc = 1.9163255161794854, scale = 1.305277180270236),
    norm(loc = 0.5895435471692753, scale = 0.4716956239592801),
    norm(loc = 0.35642887293459585, scale = 0.32591074196897313),
    norm(loc = 0.12687135373205516, scale = 0.2088557954470283),
    norm(loc = -0.05071102269302206, scale = 0.13245657228917934),
    norm(loc = -0.17053220780223977, scale = 0.14051498820792832),
    norm(loc = -0.24850747327886272, scale = 0.132687957369612)
]