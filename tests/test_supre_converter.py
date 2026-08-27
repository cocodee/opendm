from tools.convert_lerobot_supre import (
    EXPECTED_FORCE_JOINTS,
    EXPECTED_JOINTS,
    expand_supre_vector,
)


def test_supre_schema_has_15_source_joints():
    assert len(EXPECTED_JOINTS) == 15
    assert EXPECTED_JOINTS[-2:] == ["trunk_joint_1.pos", "trunk_joint_2.pos"]
    assert EXPECTED_FORCE_JOINTS[-2:] == [
        "trunk_joint_1.force",
        "trunk_joint_2.force",
    ]


def test_expand_supre_vector_inserts_zero_right_gripper_before_trunk():
    values = list(range(15))
    expanded = expand_supre_vector(values, field="action")
    assert len(expanded) == 16
    assert expanded[:13] == list(range(13))
    assert expanded[13] == 0.0
    assert expanded[14:] == [13, 14]
