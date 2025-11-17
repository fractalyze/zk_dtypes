#include "zk_dtypes/include/byinverter.h"

#include "gtest/gtest.h"

namespace zk_dtypes {
namespace {

// The cases of a prime modulus and trivial representation of both input and
// output of the inversion method. The modulus is the order of the scalar field
// of the bn254 curve
TEST(BYInverterTest, PrimeTrivial) {
  // clang-format off
  BigInt<4> modulus = *BigInt<4>::FromDecString("21888242871839275222246405745257275088548364400416034343698204186575808495617");
  // clang-format on
  BigInt<4> adjuster = BigInt<4>::One();
  BYInverter<4> inverter = BYInverter<4>(modulus, adjuster);
  BigInt<4> input, output, expected;

  // clang-format off
  input = *BigInt<4>::FromDecString("10301682245539588593700946344867822453086049145898300978024229869129994491070");
  expected = *BigInt<4>::FromDecString("19861955273495805056323792735084178237754835660372630080200899026218976035016");
  // clang-format on
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  // clang-format off
  input = *BigInt<4>::FromDecString("12652044941854844715173423049826416996530473979701897216376403046107980331138");
  expected = *BigInt<4>::FromDecString("13741189107397087982705504294857428364781781987161594150515844965152716038798");
  // clang-format on
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  // clang-format off
  input = *BigInt<4>::FromDecString("21230331365093840156003397139988613522090006176623726332211746793714384261253");
  expected = *BigInt<4>::FromDecString("19921293898917216305665589554741878428927664822386575263070110139203740591967");
  // clang-format on
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  EXPECT_FALSE(inverter.Invert(BigInt<4>::Zero(), output));
}

// The cases of a prime modulus and Montgomery representation of both input and
// output of the inversion method. The modulus is the order of the scalar field
// of the bn254 curve. The Montgomery factor equals 2²⁵⁶. For the numbers
// specified in Montgomery representation their trivial form is in the comments
TEST(BYInverterTest, PrimeMontgomery) {
  // clang-format off
  BigInt<4> modulus = *BigInt<4>::FromDecString("21888242871839275222246405745257275088548364400416034343698204186575808495617");
  BigInt<4> adjuster = *BigInt<4>::FromDecString("944936681149208446651664254269745548490766851729442924617792859073125903783");
  // clang-format on
  BYInverter<4> inverter = BYInverter<4>(modulus, adjuster);
  BigInt<4> input, output, expected;

  // clang-format off
  // 104956732223578925557223488830078362736943595779422517109872604305846127009997
  input = *BigInt<4>::FromDecString("3587950118779249390671340287316076390902306569067939221007401872613506787085");
  // 13836902468045855406793973610070160223721877582113063977376419473093071358947
  expected = *BigInt<4>::FromDecString("21148859290742718111309975246738449025500499639867444681708371655108571982666");
  // clang-format on
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  // clang-format off
  // 58137146071470818113497440987533709949510360232295259246272891334885364755793
  input = *BigInt<4>::FromDecString("5670566320610056885969457859759579419133433513714498255997485492672951983989");
  // 10012933272639613859587515636169426221587622189722693700891691228635545350648
  expected = *BigInt<4>::FromDecString("4481063342015484243022651308920441597542367033692747061403998200443241543910");
  // clang-format on
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  // clang-format off
  // 28936367257625975338385437547494218524455841456842373037716593520344969509397
  input = *BigInt<4>::FromDecString("3944075801379110538957423213814254967012269314474438285550888550560988003234");
  // 5533769493925008693547098095539268782323697506580128579981487108293743418881
  expected = *BigInt<4>::FromDecString("13708320984804885258444196338981793590913320537717878342227378760662572927772");
  // clang-format on
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  EXPECT_FALSE(inverter.Invert(BigInt<4>::Zero(), output));
}

// The cases of a composite modulus and trivial representation of both input and
// output of the inversion method. The modulus is 2 plus the order of the scalar
// field of the bn254 curve
TEST(BYInverterTest, CompositeTrivial) {
  // clang-format off
  BigInt<4> modulus = *BigInt<4>::FromDecString("21888242871839275222246405745257275088548364400416034343698204186575808495619");
  // clang-format on
  BigInt<4> adjuster = BigInt<4>::One();
  BYInverter<4> inverter = BYInverter<4>(modulus, adjuster);
  BigInt<4> input, output, expected;

  // clang-format off
  input = *BigInt<4>::FromDecString("2604009029092032058019271378824050699092064129912618878355163518774318320405");
  // clang-format on
  // Not invertible, since GCD(input, modulus) = 3
  EXPECT_FALSE(inverter.Invert(input, output));

  // clang-format off
  input = *BigInt<4>::FromDecString("6178869819981248081742625789673003349367130101292818777405002285491137113070");
  expected = *BigInt<4>::FromDecString("11809146188498084492846869194087260781155786893303570514443399828358767212457");
  // clang-format on
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  // clang-format off
  input = *BigInt<4>::FromDecString("11889004628415053025680830661657644850370312921268517007619583090643890709598");
  expected = *BigInt<4>::FromDecString("9547272606271509701441668321595525482503472735300288327005468472209166908277");
  // clang-format on
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  EXPECT_FALSE(inverter.Invert(BigInt<4>::Zero(), output));
}

// The cases of a composite modulus and Montgomery representation of both
// input and output of the inversion method. The modulus is 2 plus the order
// of the scalar field of the bn254 curve. The Montgomery factor equals 2²⁵⁶.
// For the numbers specified in Montgomery representation their trivial form
// is in the comments
TEST(BYInverterTest, CompositeMontgomery) {
  // clang-format off
  BigInt<4> modulus = *BigInt<4>::FromDecString("21888242871839275222246405745257275088548364400416034343698204186575808495619");
  BigInt<4> adjuster = *BigInt<4>::FromDecString("1571482956516473071827433438586836372334964131807058629613396720502838675869");
  // clang-format on
  BYInverter<4> inverter = BYInverter<4>(modulus, adjuster);
  BigInt<4> input, output, expected;

  // clang-format off
  // 11865630156646177845488966194957438762902332631956652259274124517358851833759
  input = *BigInt<4>::FromDecString("8782016184547047265226504283506537510714797770456954233217641807032605305262");
  // clang-format on
  // Not invertible, since GCD(input, modulus) = 3
  EXPECT_FALSE(inverter.Invert(input, output));

  // clang-format off
  // 13626187621506977415144012068495580987161044579599980007720002490790855281118
  input = *BigInt<4>::FromDecString("11222801578780002660116334909264887192550017735727671483478588640885635141103");
  // 15569609600917656079795063009991863730429494071872866284038456731797547021054
  expected = *BigInt<4>::FromDecString("18386134469083625139458350071877715518509144165860762127278364281897433970018");
  // clang-format on
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  // clang-format off
  // 4652503444426046901695366961113196520526566962552761000864601049682876484005
  input = *BigInt<4>::FromDecString("5910683103293609884541455561942593321471963402548599885686194071431121247969");
  // 4735762809510279031273699449377321073262538802381459266345256675192986276306
  expected = *BigInt<4>::FromDecString("9624059289414117296690079084891323779865370094538843278357651910292971494292");
  // clang-format on
  EXPECT_TRUE(inverter.Invert(input, output));
  EXPECT_EQ(output, expected);

  EXPECT_FALSE(inverter.Invert(BigInt<4>::Zero(), output));
}

}  // namespace
}  // namespace zk_dtypes
