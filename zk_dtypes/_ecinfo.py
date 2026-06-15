# Copyright 2026 The zk_dtypes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Overload of numpy.ecinfo to handle EC point dtypes defined in zk_dtypes."""

from zk_dtypes._zk_dtypes_ext import bn254_sf
from zk_dtypes._zk_dtypes_ext import bn254_sf_mont
from zk_dtypes._zk_dtypes_ext import bn254_g1_affine
from zk_dtypes._zk_dtypes_ext import bn254_g1_affine_mont
from zk_dtypes._zk_dtypes_ext import bn254_g1_jacobian
from zk_dtypes._zk_dtypes_ext import bn254_g1_jacobian_mont
from zk_dtypes._zk_dtypes_ext import bn254_g1_xyzz
from zk_dtypes._zk_dtypes_ext import bn254_g1_xyzz_mont
from zk_dtypes._zk_dtypes_ext import bn254_g2_affine
from zk_dtypes._zk_dtypes_ext import bn254_g2_affine_mont
from zk_dtypes._zk_dtypes_ext import bn254_g2_jacobian
from zk_dtypes._zk_dtypes_ext import bn254_g2_jacobian_mont
from zk_dtypes._zk_dtypes_ext import bn254_g2_xyzz
from zk_dtypes._zk_dtypes_ext import bn254_g2_xyzz_mont
from zk_dtypes._zk_dtypes_ext import mnt4_298_sf
from zk_dtypes._zk_dtypes_ext import mnt4_298_sf_mont
from zk_dtypes._zk_dtypes_ext import mnt4_298_g1_affine
from zk_dtypes._zk_dtypes_ext import mnt4_298_g1_affine_mont
from zk_dtypes._zk_dtypes_ext import mnt4_298_g1_jacobian
from zk_dtypes._zk_dtypes_ext import mnt4_298_g1_jacobian_mont
from zk_dtypes._zk_dtypes_ext import mnt4_298_g1_xyzz
from zk_dtypes._zk_dtypes_ext import mnt4_298_g1_xyzz_mont
from zk_dtypes._zk_dtypes_ext import mnt4_298_g2_affine
from zk_dtypes._zk_dtypes_ext import mnt4_298_g2_affine_mont
from zk_dtypes._zk_dtypes_ext import mnt4_298_g2_jacobian
from zk_dtypes._zk_dtypes_ext import mnt4_298_g2_jacobian_mont
from zk_dtypes._zk_dtypes_ext import mnt4_298_g2_xyzz
from zk_dtypes._zk_dtypes_ext import mnt4_298_g2_xyzz_mont
from zk_dtypes._zk_dtypes_ext import mnt6_298_sf
from zk_dtypes._zk_dtypes_ext import mnt6_298_sf_mont
from zk_dtypes._zk_dtypes_ext import mnt6_298_g1_affine
from zk_dtypes._zk_dtypes_ext import mnt6_298_g1_affine_mont
from zk_dtypes._zk_dtypes_ext import mnt6_298_g1_jacobian
from zk_dtypes._zk_dtypes_ext import mnt6_298_g1_jacobian_mont
from zk_dtypes._zk_dtypes_ext import mnt6_298_g1_xyzz
from zk_dtypes._zk_dtypes_ext import mnt6_298_g1_xyzz_mont
from zk_dtypes._zk_dtypes_ext import mnt6_298_g2_affine
from zk_dtypes._zk_dtypes_ext import mnt6_298_g2_affine_mont
from zk_dtypes._zk_dtypes_ext import mnt6_298_g2_jacobian
from zk_dtypes._zk_dtypes_ext import mnt6_298_g2_jacobian_mont
from zk_dtypes._zk_dtypes_ext import mnt6_298_g2_xyzz
from zk_dtypes._zk_dtypes_ext import mnt6_298_g2_xyzz_mont

import numpy as np

_bn254_sf_dtype = np.dtype(bn254_sf)
_bn254_sf_mont_dtype = np.dtype(bn254_sf_mont)
_mnt4_298_sf_dtype = np.dtype(mnt4_298_sf)
_mnt4_298_sf_mont_dtype = np.dtype(mnt4_298_sf_mont)
_mnt6_298_sf_dtype = np.dtype(mnt6_298_sf)
_mnt6_298_sf_mont_dtype = np.dtype(mnt6_298_sf_mont)

# ---------------------------------------------------------------------------
# bn254 curve parameters (y² = x³ + ax + b). G2 lives over Fp² = Fq[u]/(u²-β).
# ---------------------------------------------------------------------------
_BN254_G1_A = 0
_BN254_G1_B = 3
_BN254_G1_GX = 1
_BN254_G1_GY = 2

# G1 Montgomery form: each value * R mod Fq
_BN254_G1_A_MONT = 0
_BN254_G1_B_MONT = 19052624634359457937016868847204597229365286637454337178037183604060995791063
_BN254_G1_GX_MONT = (
    6350874878119819312338956282401532409788428879151445726012394534686998597021
)
_BN254_G1_GY_MONT = 12701749756239638624677912564803064819576857758302891452024789069373997194042

# G2 standard form: Fp2 elements [c₀, c₁] where val = c₀ + c₁ * u
_BN254_G2_A = [0, 0]
_BN254_G2_B = [
    19485874751759354771024239261021720505790618469301721065564631296452457478373,
    266929791119991161246907387137283842545076965332900288569378510910307636690,
]
_BN254_G2_GX = [
    10857046999023057135944570762232829481370756359578518086990519993285655852781,
    11559732032986387107991004021392285783925812861821192530917403151452391805634,
]
_BN254_G2_GY = [
    8495653923123431417604973247489272438418190587263600148770280649306958101930,
    4082367875863433681332203403145435568316851327593401208105741076214120093531,
]
_BN254_G2_NON_RESIDUE = 21888242871839275222246405745257275088696311157297823662689037894645226208582

# G2 Montgomery form: each Fp² component * R mod Fq
_BN254_G2_A_MONT = [0, 0]
_BN254_G2_B_MONT = [
    16772280239760917788496391897731603718812008455956943122563801666366297604776,
    568440292453150825972223760836185707764922522371208948902804025364325400423,
]
_BN254_G2_GX_MONT = [
    11461925177900819176832270005713103520318409907105193817603008068482420711462,
    9496696083199853777875401760424613833161720860855390556979200160215841136960,
]
_BN254_G2_GY_MONT = [
    18540402224736191443939503902445128293982106376239432540843647066670759668214,
    6170940445994484564222204938066213705353407449799250191249554538140978927342,
]
_BN254_G2_NON_RESIDUE_MONT = 15537367993719455909907449462855742678907882278146377936676643359958227611562

# ---------------------------------------------------------------------------
# MNT4-298 curve parameters (ark-mnt4-298). G1: y² = x³ + 2x + b over Fq.
# G2 over Fp² = Fq[u]/(u²-17); twist a = (34, 0), b = (0, b·17). 320-bit field.
# Montgomery form = value * R mod p, R = 2³²⁰.
# ---------------------------------------------------------------------------
_MNT4_298_G1_A = 2
_MNT4_298_G1_B = 423894536526684178289416011533888240029318103673896002803341544124054745019340795360841685
_MNT4_298_G1_GX = 60760244141852568949126569781626075788424196370144486719385562369396875346601926534016838
_MNT4_298_G1_GY = 363732850702582978263902770815145784459747722357071843971107674179038674942891694705904306

_MNT4_298_G1_A_MONT = 446729296652562829877603410718446059103655029145798501270518660890454746243743614887200952
_MNT4_298_G1_B_MONT = 179877917358261777753314219897940120444991346955733434661847233718565124864239546371263281
_MNT4_298_G1_GX_MONT = 354680644509934968175469258381357629814402639583090735032750513901318014671205221931808948
_MNT4_298_G1_GY_MONT = 445280841095252623635668290173967960852066584341969329684246826090979024898521426833907837

_MNT4_298_G2_A = [34, 0]
_MNT4_298_G2_B = [
    0,
    67372828414711144619833451280373307321534573815811166723479321465776723059456513877937430,
]
_MNT4_298_G2_GX = [
    438374926219350099854919100077809681842783509163790991847867546339851681564223481322252708,
    37620953615500480110935514360923278605464476459712393277679280819942849043649216370485641,
]
_MNT4_298_G2_GY = [
    37437409008528968268352521034936931842973546441370663118543015118291998305624025037512482,
    424621479598893882672393190337420680597584695892317197646113820787463109735345923009077489,
]
_MNT4_298_G2_NON_RESIDUE = 17

_MNT4_298_G2_A_MONT = [
    455563750554648221619019237417856231585262306838153640665490306494576743874304445826044969,
    0,
]
_MNT4_298_G2_B_MONT = [
    0,
    202390878074882267286246240346691338294103622791300036878072201758345545784337485408927291,
]
_MNT4_298_G2_GX_MONT = [
    187432131579426194088031088405014791977840906396261280983391303286363730912415292160457714,
    266760376307210572592346313524809001086973414552101896394659348925629059473236215626936176,
]
_MNT4_298_G2_GY_MONT = [
    306057346885993003434071827587844987524397598107209712539417326681795274602240783738183828,
    250975396713672639336952031314618276534945250112062301003586398268356393471632083123571796,
]
_MNT4_298_G2_NON_RESIDUE_MONT = 465743018361954773686184243535452341565193593040424183030522717535393503346130123154901525

# ---------------------------------------------------------------------------
# MNT6-298 curve parameters (ark-mnt6-298). G1: y² = x³ + 11x + b over Fq.
# G2 over Fp³ = Fq[u]/(u³-5); twist a = (0, 0, 11), b = (5·b, 0, 0). 320-bit
# field (the MNT4 cycle: MNT6 Fq = MNT4 Fr). Montgomery form = value·R mod p.
# ---------------------------------------------------------------------------
_MNT6_298_G1_A = 11
_MNT6_298_G1_B = 106700080510851735677967319632585352256454251201367587890185989362936000262606668469523074
_MNT6_298_G1_GX = 336685752883082228109289846353937104185698209371404178342968838739115829740084426881123453
_MNT6_298_G1_GY = 402596290139780989709332707716568920777622032073762749862342374583908837063963736098549800

_MNT6_298_G1_A_MONT = 77399700742788935560072510686211067378536588283599049747225344898379629001906254049619951
_MNT6_298_G1_B_MONT = 117863729424218755314889477576976072108240353392280102964914008159214280612609743972627259
_MNT6_298_G1_GX_MONT = 66803446633995354443436432086785857429890759773793771766712172085558096199224671983246628
_MNT6_298_G1_GY_MONT = 391921278559921373872478291534168668984317278471439419727775904969858842965515216515760050

_MNT6_298_G2_A = [0, 0, 11]
_MNT6_298_G2_B = [
    57578116384997352636487348509878309737146377454014423897662211075515354005624851787652233,
    0,
    0,
]
_MNT6_298_G2_GX = [
    421456435772811846256826561593908322288509115489119907560382401870203318738334702321297427,
    103072927438548502463527009961344915021167584706439945404959058962657261178393635706405114,
    143029172143731852627002926324735183809768363301149009204849580478324784395590388826052558,
]
_MNT6_298_G2_GY = [
    464673596668689463130099227575639512541218133445388869383893594087634649237515554342751377,
    100642907501977375184575075967118071807821117960152743335603284583254620685343989304941678,
    123019855502969896026940545715841181300275180157288044663051565390506010149881373807142903,
]
_MNT6_298_G2_NON_RESIDUE = 5

_MNT6_298_G2_A_MONT = [
    0,
    0,
    77399700742788935560072510686211067378536588283599049747225344898379629001906254049619951,
]
_MNT6_298_G2_B_MONT = [
    113396360951832450821098138231831908996076888408576999271302305056906755755640229303173158,
    0,
    0,
]
_MNT6_298_G2_GX_MONT = [
    288208926410660646513785090852407832338233401063728316230268857502404730449173144485043879,
    351381625308823176868807127486138634595116949883628439586787050017946781674344146387034416,
    127672217964183870707978709919069290404704306329671129014740702808526116370651618768708081,
]
_MNT6_298_G2_GY_MONT = [
    149587757998830678715843103866538208153562829471370761178971527424848110063377843185155167,
    105322739714326088692821876215799163825059720357780792387795512794429173527988284429257703,
    359945110800750513286064290068019448559925840421393865155861545034129653722834631881439455,
]
_MNT6_298_G2_NON_RESIDUE_MONT = 164978669292884423187310027490018244684368870643315072308720902882672007902886976538908106

# Per-(curve, group, is_montgomery) curve coefficient + generator tables.
_CURVE_PARAMS = {
    ('bn254', 'g1', False): dict(
        a=_BN254_G1_A,
        b=_BN254_G1_B,
        gx=_BN254_G1_GX,
        gy=_BN254_G1_GY,
        non_residue=None,
    ),
    ('bn254', 'g1', True): dict(
        a=_BN254_G1_A_MONT,
        b=_BN254_G1_B_MONT,
        gx=_BN254_G1_GX_MONT,
        gy=_BN254_G1_GY_MONT,
        non_residue=None,
    ),
    ('bn254', 'g2', False): dict(
        a=_BN254_G2_A,
        b=_BN254_G2_B,
        gx=_BN254_G2_GX,
        gy=_BN254_G2_GY,
        non_residue=_BN254_G2_NON_RESIDUE,
    ),
    ('bn254', 'g2', True): dict(
        a=_BN254_G2_A_MONT,
        b=_BN254_G2_B_MONT,
        gx=_BN254_G2_GX_MONT,
        gy=_BN254_G2_GY_MONT,
        non_residue=_BN254_G2_NON_RESIDUE_MONT,
    ),
    ('mnt4_298', 'g1', False): dict(
        a=_MNT4_298_G1_A,
        b=_MNT4_298_G1_B,
        gx=_MNT4_298_G1_GX,
        gy=_MNT4_298_G1_GY,
        non_residue=None,
    ),
    ('mnt4_298', 'g1', True): dict(
        a=_MNT4_298_G1_A_MONT,
        b=_MNT4_298_G1_B_MONT,
        gx=_MNT4_298_G1_GX_MONT,
        gy=_MNT4_298_G1_GY_MONT,
        non_residue=None,
    ),
    ('mnt4_298', 'g2', False): dict(
        a=_MNT4_298_G2_A,
        b=_MNT4_298_G2_B,
        gx=_MNT4_298_G2_GX,
        gy=_MNT4_298_G2_GY,
        non_residue=_MNT4_298_G2_NON_RESIDUE,
    ),
    ('mnt4_298', 'g2', True): dict(
        a=_MNT4_298_G2_A_MONT,
        b=_MNT4_298_G2_B_MONT,
        gx=_MNT4_298_G2_GX_MONT,
        gy=_MNT4_298_G2_GY_MONT,
        non_residue=_MNT4_298_G2_NON_RESIDUE_MONT,
    ),
    ('mnt6_298', 'g1', False): dict(
        a=_MNT6_298_G1_A,
        b=_MNT6_298_G1_B,
        gx=_MNT6_298_G1_GX,
        gy=_MNT6_298_G1_GY,
        non_residue=None,
    ),
    ('mnt6_298', 'g1', True): dict(
        a=_MNT6_298_G1_A_MONT,
        b=_MNT6_298_G1_B_MONT,
        gx=_MNT6_298_G1_GX_MONT,
        gy=_MNT6_298_G1_GY_MONT,
        non_residue=None,
    ),
    ('mnt6_298', 'g2', False): dict(
        a=_MNT6_298_G2_A,
        b=_MNT6_298_G2_B,
        gx=_MNT6_298_G2_GX,
        gy=_MNT6_298_G2_GY,
        non_residue=_MNT6_298_G2_NON_RESIDUE,
    ),
    ('mnt6_298', 'g2', True): dict(
        a=_MNT6_298_G2_A_MONT,
        b=_MNT6_298_G2_B_MONT,
        gx=_MNT6_298_G2_GX_MONT,
        gy=_MNT6_298_G2_GY_MONT,
        non_residue=_MNT6_298_G2_NON_RESIDUE_MONT,
    ),
}

# (group, repr, num_coords). G1 coords are in the base field; G2 coords are in
# the degree-`g2_ext` extension (Fp2 for bn254/mnt4, Fp3 for mnt6).
_EC_LAYOUT = [
    ('g1', 'affine', 2),
    ('g1', 'jacobian', 3),
    ('g1', 'xyzz', 4),
    ('g2', 'affine', 2),
    ('g2', 'jacobian', 3),
    ('g2', 'xyzz', 4),
]


def _build_meta(
    curve, field_bits, g2_ext, scalar_std, scalar_mont, dtype_table
):
  """Maps each registered point dtype -> (curve, group, repr, is_mont,
  storage_bits, base_field_dtype)."""
  meta = {}
  for group, repr_, num_coords in _EC_LAYOUT:
    ext = 1 if group == 'g1' else g2_ext
    bits = num_coords * ext * field_bits
    for is_mont in (False, True):
      dt = np.dtype(dtype_table[(group, repr_, is_mont)])
      base = scalar_mont if is_mont else scalar_std
      meta[dt] = (curve, group, repr_, is_mont, bits, base)
  return meta


_EC_DTYPE_META = {}
_EC_DTYPE_META.update(
    _build_meta(
        'bn254',
        256,
        2,
        _bn254_sf_dtype,
        _bn254_sf_mont_dtype,
        {
            ('g1', 'affine', False): bn254_g1_affine,
            ('g1', 'affine', True): bn254_g1_affine_mont,
            ('g1', 'jacobian', False): bn254_g1_jacobian,
            ('g1', 'jacobian', True): bn254_g1_jacobian_mont,
            ('g1', 'xyzz', False): bn254_g1_xyzz,
            ('g1', 'xyzz', True): bn254_g1_xyzz_mont,
            ('g2', 'affine', False): bn254_g2_affine,
            ('g2', 'affine', True): bn254_g2_affine_mont,
            ('g2', 'jacobian', False): bn254_g2_jacobian,
            ('g2', 'jacobian', True): bn254_g2_jacobian_mont,
            ('g2', 'xyzz', False): bn254_g2_xyzz,
            ('g2', 'xyzz', True): bn254_g2_xyzz_mont,
        },
    )
)
_EC_DTYPE_META.update(
    _build_meta(
        'mnt4_298',
        320,
        2,
        _mnt4_298_sf_dtype,
        _mnt4_298_sf_mont_dtype,
        {
            ('g1', 'affine', False): mnt4_298_g1_affine,
            ('g1', 'affine', True): mnt4_298_g1_affine_mont,
            ('g1', 'jacobian', False): mnt4_298_g1_jacobian,
            ('g1', 'jacobian', True): mnt4_298_g1_jacobian_mont,
            ('g1', 'xyzz', False): mnt4_298_g1_xyzz,
            ('g1', 'xyzz', True): mnt4_298_g1_xyzz_mont,
            ('g2', 'affine', False): mnt4_298_g2_affine,
            ('g2', 'affine', True): mnt4_298_g2_affine_mont,
            ('g2', 'jacobian', False): mnt4_298_g2_jacobian,
            ('g2', 'jacobian', True): mnt4_298_g2_jacobian_mont,
            ('g2', 'xyzz', False): mnt4_298_g2_xyzz,
            ('g2', 'xyzz', True): mnt4_298_g2_xyzz_mont,
        },
    )
)
_EC_DTYPE_META.update(
    _build_meta(
        'mnt6_298',
        320,
        3,
        _mnt6_298_sf_dtype,
        _mnt6_298_sf_mont_dtype,
        {
            ('g1', 'affine', False): mnt6_298_g1_affine,
            ('g1', 'affine', True): mnt6_298_g1_affine_mont,
            ('g1', 'jacobian', False): mnt6_298_g1_jacobian,
            ('g1', 'jacobian', True): mnt6_298_g1_jacobian_mont,
            ('g1', 'xyzz', False): mnt6_298_g1_xyzz,
            ('g1', 'xyzz', True): mnt6_298_g1_xyzz_mont,
            ('g2', 'affine', False): mnt6_298_g2_affine,
            ('g2', 'affine', True): mnt6_298_g2_affine_mont,
            ('g2', 'jacobian', False): mnt6_298_g2_jacobian,
            ('g2', 'jacobian', True): mnt6_298_g2_jacobian_mont,
            ('g2', 'xyzz', False): mnt6_298_g2_xyzz,
            ('g2', 'xyzz', True): mnt6_298_g2_xyzz_mont,
        },
    )
)


class ecinfo:  # pylint: disable=invalid-name,missing-class-docstring
  base_field_dtype: np.dtype
  storage_bits: int
  point_repr: str  # 'affine', 'jacobian', or 'xyzz'
  curve_group: str  # 'g1' or 'g2'
  is_montgomery: bool
  a: int | list[int]  # curve coefficient a in y² = x³ + ax + b
  b: int | list[int]  # curve coefficient b
  gx: int | list[int]  # generator x-coordinate
  gy: int | list[int]  # generator y-coordinate
  non_residue: int | None  # Fp²/Fp³ non-residue (None for G1, int for G2)
  dtype: np.dtype

  def __init__(self, ec_type):
    ec_type = np.dtype(ec_type)
    meta = _EC_DTYPE_META.get(ec_type)
    if meta is None:
      raise ValueError(f'Unknown elliptic curve point type: {ec_type}')
    curve, group, point_repr, is_montgomery, storage_bits, base_field_dtype = (
        meta
    )

    self.dtype = ec_type
    self.curve_group = group
    self.point_repr = point_repr
    self.is_montgomery = is_montgomery
    self.storage_bits = storage_bits
    self.base_field_dtype = base_field_dtype

    params = _CURVE_PARAMS[(curve, group, is_montgomery)]
    self.a = params['a']
    self.b = params['b']
    self.gx = params['gx']
    self.gy = params['gy']
    self.non_residue = params['non_residue']

  def __repr__(self):
    return f'ecinfo(curve_group={self.curve_group}, point_repr={self.point_repr}, dtype={self.dtype})'

  def __str__(self):
    return repr(self)
