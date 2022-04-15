# %%
import datasets
from datasets import load_dataset, load_metric, Audio, load_from_disk
train_mapped = load_from_disk("train220k_mapped")
troubled_idx = [
    170,
    474,
    778,
    1082,
    1385,
    1688,
    1991,
    2294,
    2597,
    2900,
    3203,
    3506,
    3809,
    4111,
    4413,
    4715,
    5017,
    5319,
    5620,
    5921,
    6222,
    6523,
    6823,
    7123,
    7422,
    7721,
    8020,
    8319,
    8618,
    8917,
    9216,
    9515,
    9814,
    10113,
    10412,
    10711,
    11010,
    11309,
    11608,
    11907,
    12206,
    12505,
    12804,
    13103,
    13401,
    13699,
    13997,
    14295,
    14593,
    14891,
    15189,
    15487,
    15785,
    16083,
    16381,
    16679,
    16977,
    17275,
    17573,
    17871,
    18169,
    18467,
    18765,
    19063,
    19361,
    19659,
    19957,
    20255,
    20553,
    20850,
    21147,
    21444,
    21741,
    22038,
    22335,
    22632,
    22929,
    23226,
    23523,
    23820,
    24117,
    24414,
    24711,
    25008,
    25305,
    25602,
    25899,
    26196,
    26493,
    26790,
    27087,
    27384,
    27681,
    27978,
    28275,
    28572,
    28869,
    29166,
    29463,
    29760,
    30057,
    30354,
    30651,
    30948,
    31245,
    31542,
    31839,
    32136,
    32433,
    32730,
    33027,
    33324,
    33621,
    33918,
    34215,
    34512,
    34809,
    35106,
    35402,
    35698,
    35994,
    36290,
    36586,
    36882,
    37178,
    37474,
    37770,
    38066,
    38362,
    38658,
    38953,
    39248,
    39543,
    39838,
    40133,
    40428,
    40723,
    41018,
    41313,
    41608,
    41903,
    42198,
    42493,
    42788,
    43083,
    43378,
    43673,
    43968,
    44263,
    44558,
    44853,
    45148,
    45443,
    45738,
    46033,
    46328,
    46623,
    46918,
    47213,
    47508,
    47803,
    48098,
    48393,
    48688,
    48983,
    49278,
    49573,
    49868,
    50163,
    50458,
    50753,
    51048,
    51343,
    51638,
    51933,
    52228,
    52523,
    52818,
    53113,
    53408,
    53703,
    53998,
    54293,
    54588,
    54883,
    55178,
    55473,
    55768,
    56063,
    56358,
    56653,
    56948,
    57242,
    57536,
    57830,
    58124,
    58418,
    58712,
    59006,
    59300,
    59594,
    59888,
    60182,
    60476,
    60770,
    61064,
    61358,
    61652,
    61946,
    62240,
    62534,
    62828,
    63122,
    63416,
    63710,
    64004,
    64298,
    64592,
    64885,
    65178,
    65471,
    65764,
    66057,
    66350,
    66643,
    66936,
    67229,
    67522,
    67815,
    68108,
    68401,
    68694,
    68987,
    69280,
    69573,
    69866,
    70159,
    70452,
    70745,
    71037,
    71329,
    71621,
    71913,
    72205,
    72497,
    72789,
    73081,
    73373,
    73665,
    73957,
    74249,
    74541,
    74833,
    75125,
    75417,
    75709,
    76001,
    76293,
    76585,
    76877,
    77169,
    77461,
    77753,
    78045,
    78337,
    78629,
    78921,
    79213,
    79505,
    79797,
    80089,
    80381,
    80673,
    80965,
    81257,
    81549,
    81840,
    82131,
    82422,
    82713,
    83004,
    83295,
    83586,
    83877,
    84168,
    84459,
    84750,
    85041,
    85332,
    85623,
    85914,
    86205,
    86496,
    86787,
    87078,
    87369,
    87660,
    87951,
    88242,
    88533,
    88824,
    89115,
    89406,
    89697,
    89988,
    90279,
    90570,
    90861,
    91152,
    91443,
    91734,
    92025,
    92316,
    92607,
    92898,
    93189,
    93480,
    93771,
    94062,
    94353,
    94644,
    94935,
    95226,
    95516,
    95806,
    96096,
    96386,
    96676,
    96966,
    97255,
    97544,
    97833,
    98122,
    98411,
    98700,
    98989,
    99278,
    99567,
    99856,
    100145,
    100434,
    100723,
    101012,
    101301,
    101590,
    101879,
    102168,
    102457,
    102746,
    103035,
    103324,
    103613,
    103902,
    104191,
    104480,
    104769,
    105058,
    105347,
    105635,
    105923,
    106211,
    106499,
    106787,
    107075,
    107363,
    107651,
    107939,
    108227,
    108514,
    108801,
    109088,
    109375,
    109662,
    109949,
    110236,
    110523,
    110810,
    111097,
    111384,
    111671,
    111958,
    112245,
    112532,
    112819,
    113106,
    113393,
    113680,
    113967,
    114254,
    114541,
    114828,
    115115,
    115402,
    115689,
    115976,
    116263,
    116550,
    116836,
    117122,
    117408,
    117694,
    117980,
    118266,
    118551,
    118836,
    119121,
    119406,
    119691,
    119976,
    120261,
    120546,
    120831,
    121116,
    121401,
    121686,
    121971,
    122256,
    122541,
    122826,
    123111,
    123396,
    123681,
    123966,
    124251,
    124536,
    124821,
    125106,
    125391,
    125676,
    125961,
    126246,
    126531,
    126816,
    127101,
    127386,
    127671,
    127956,
    128241,
    128526,
    128811,
    129096,
    129380,
    129664,
    129948,
    130232,
    130516,
    130800,
    131084,
    131368,
    131652,
    131936,
    132220,
    132504,
    132788,
    133072,
    133356,
    133640,
    133924,
    134208,
    134492,
    134776,
    135060,
    135344,
    135628,
    135912,
    136196,
    136480,
    136764,
    137048,
    137332,
    137616,
    137900,
    138184,
    138468,
    138752,
    139036,
    139320,
    139604,
    139888,
    140172,
    140456,
    140740,
    141024,
    141308,
    141592,
    141876,
    142160,
    142444,
    142728,
    143012,
    143296,
    143579,
    143862,
    144145,
    144428,
    144711,
    144994,
    145277,
    145560,
    145843,
    146126,
    146409,
    146692,
    146975,
    147258,
    147541,
    147824,
    148107,
    148390,
    148673,
    148956,
    149239,
    149522,
    149805,
    150088,
    150371,
    150654,
    150937,
    151220,
    151502,
    151784,
    152066,
    152348,
    152630,
    152912,
    153194,
    153476,
    153758,
    154040,
    154322,
    154604,
    154886,
    155168,
    155450,
    155732,
    156014,
    156296,
    156578,
    156860,
    157142,
    157424,
    157706,
    157988,
    158270,
    158552,
    158834,
    159116,
    159398,
    159680,
    159962,
    160244,
    160526,
    160808,
    161090,
    161372,
    161654,
    161936,
    162218,
    162500,
    162782,
    163064,
    163346,
    163628,
    163910,
    164192,
    164474,
    164756,
    165038,
    165320,
    165602,
    165884,
    166166,
    166448,
    166730,
    167012,
    167294,
    167576,
    167858,
    168140,
    168422,
    168704,
    168986,
    169268,
    169550,
    169832,
    170114,
    170396,
    170678,
    170960,
    171242,
    171524,
    171806,
    172088,
    172370,
    172652,
    172934,
    173216,
    173498,
    173780,
    174062,
    174344,
    174626,
    174908,
    175190,
    175472,
    175754,
    176036,
    176318,
    176600,
    176882,
    177164,
    177446,
    177728,
    178009,
    178290,
    178571,
    178852,
    179133,
    179414,
    179695,
    179976,
    180257,
    180538,
    180819,
    181100,
    181381,
    181662,
    181943,
    182224,
    182505,
    182786,
    183067,
    183348,
    183629,
    183910,
    184191,
    184472,
    184753,
    185034,
    185315,
    185596,
    185877,
    186158,
    186439,
    186720,
    187001,
    187282,
    187563,
    187844,
    188125,
    188406,
    188687,
    188968,
    189249,
    189530,
    189811,
    190092,
    190373,
    190654,
    190935,
    191216,
    191497,
    191777,
    192057,
    192337,
    192617,
    192897,
    193177,
    193457,
    193737,
    194017,
    194297,
    194577,
    194857,
    195137,
    195417,
    195697,
    195977,
    196257,
    196537,
    196817,
    197097,
    197377,
    197657,
    197937,
    198217,
    198497,
    198777,
    199057,
    199337,
    199617,
    199897,
    200177,
    200457,
    200737,
    201017,
    201297,
    201577,
    201857,
    202137,
    202417,
    202697,
    202977,
    203257,
    203537,
    203817,
    204097,
    204377,
    204657,
    204937,
    205217,
    205497,
    205777,
    206057,
    206337,
    206617,
    206897,
    207177,
    207457,
    207737,
    208017,
    208297,
    208577,
    208857,
    209137,
    209417,
    209697,
    209977,
    210257,
    210537,
    210817,
    211097,
    211377,
    211657,
    211937,
    212217,
    212497,
    212777,
    213057,
    213337,
    213617,
    213897,
    214177,
    214457,
    214737,
    215017,
    215296,
    215575,
    215854,
    216133,
    216412,
    216691,
    216970,
    217249,
    217528,
    217807,
    218086,
    218365,
    218644,
    218923,
    219202,
    219481,
    219760,
    220039]

test_mapped= load_from_disk("test_mapped")
print(f"{len(train_mapped)=}")
print("Found empties: ")
empties = []
for i in range(len(train_mapped)):
    if len(train_mapped[i]["input_values"]) == 0:
        print(i)
        empties.append(i)
