from numpy import array, arange

weights = array(
    (
        1.0000000000000004,
        1.0000000000000038,
        1.0000000000000182,
        1.0000000000000064,
        0.9999999999999962,
        1.0000000000000053,
        1.0000000000000222,
        1.0000000000000113,
        0.8912112036083922,
        0.8591167563965603,
        0.891211203608377,
        0.8912112036084067,
        0.8591167563965719,
        0.8912112036083736,
        0.8912112036084004,
        0.8591167563965604,
        0.8912112036083729,
        0.8912112036083879,
        0.8591167563965556,
        0.8912112036083712,
        0.8912112036084329,
        0.8591167563965191,
        0.8912112036084098,
        0.8912112036084323,
        0.8591167563965256,
        0.8912112036084059,
        0.8912112036084165,
        0.8591167563965465,
        0.891211203608397,
        0.8912112036084232,
        0.8591167563965371,
        0.8912112036083985,
        0.8912112036083935,
        0.8591167563965474,
        0.8912112036083979,
        0.8912112036084096,
        0.8591167563965341,
        0.8912112036084131,
        0.8912112036083994,
        0.8591167563965553,
        0.8912112036083869,
        0.8912112036084061,
        0.8591167563965514,
        0.8912112036083917,
        0.7622595264192012,
        0.7186651735399952,
        0.7622595264192108,
        0.7186651735400831,
        0.6712724315919532,
        0.7186651735400142,
        0.7622595264191386,
        0.7186651735400778,
        0.7622595264191658,
        0.7622595264191585,
        0.7186651735400396,
        0.7622595264191763,
        0.7186651735401408,
        0.6712724315918481,
        0.7186651735400713,
        0.7622595264191132,
        0.7186651735401441,
        0.7622595264191405,
        0.7622595264191835,
        0.7186651735400215,
        0.7622595264191755,
        0.718665173540074,
        0.6712724315918527,
        0.7186651735400587,
        0.76225952641914,
        0.7186651735401088,
        0.7622595264191026,
        0.762259526419186,
        0.718665173539988,
        0.7622595264191874,
        0.718665173540024,
        0.671272431591901,
        0.7186651735400487,
        0.7622595264191483,
        0.7186651735400567,
        0.7622595264191343,
        0.7622595264191981,
        0.7186651735400303,
        0.7622595264191959,
        0.7186651735400273,
        0.6712724315919835,
        0.7186651735399883,
        0.7622595264191977,
        0.7186651735400158,
        0.7622595264191645,
        0.7622595264191123,
        0.7186651735401346,
        0.7622595264191119,
        0.7186651735401413,
        0.6712724315917651,
        0.7186651735401233,
        0.7622595264191341,
        0.718665173540117,
        0.7622595264191148,
        0.6131449684323388,
        0.5580507098859488,
        0.6131449684323087,
        0.558050709885957,
        0.499158062270623,
        0.5580507098860268,
        0.613144968432233,
        0.5580507098859713,
        0.6131449684321943,
        0.5580507098858419,
        0.49915806227084414,
        0.5580507098857239,
        0.4991580622706119,
        0.43646702558565864,
        0.49915806227082715,
        0.5580507098859069,
        0.49915806227063975,
        0.5580507098858827,
        0.6131449684325461,
        0.5580507098855338,
        0.613144968432479,
        0.5580507098856315,
        0.4991580622710732,
        0.5580507098856855,
        0.6131449684324657,
        0.5580507098857228,
        0.613144968432394,
    )
)


points = array(
    [
        -0.5773502588272095,
        -0.5773502588272095,
        -0.5773502588272095,
        0.5773502588272095,
        -0.5773502588272095,
        -0.5773502588272095,
        0.5773502588272095,
        0.5773502588272095,
        -0.5773502588272095,
        -0.5773502588272095,
        0.5773502588272095,
        -0.5773502588272095,
        -0.5773502588272095,
        -0.5773502588272095,
        0.5773502588272095,
        0.5773502588272095,
        -0.5773502588272095,
        0.5773502588272095,
        0.5773502588272095,
        0.5773502588272095,
        0.5773502588272095,
        -0.5773502588272095,
        0.5773502588272095,
        0.5773502588272095,
        -0.3128761947154999,
        -0.7095873355865479,
        -0.7095873355865479,
        1.573255109901025e-14,
        -0.7540208101272583,
        -0.7540208101272583,
        0.3128761947154999,
        -0.7095873355865479,
        -0.7095873355865479,
        0.7095873355865479,
        -0.3128761947154999,
        -0.7095873355865479,
        0.7540208101272583,
        2.990022576862712e-14,
        -0.7540208101272583,
        0.7095873355865479,
        0.3128761947154999,
        -0.7095873355865479,
        -0.3128761947154999,
        0.7095873355865479,
        -0.7095873355865479,
        2.731565719601113e-14,
        0.7540208101272583,
        -0.7540208101272583,
        0.3128761947154999,
        0.7095873355865479,
        -0.7095873355865479,
        -0.7095873355865479,
        -0.3128761947154999,
        -0.7095873355865479,
        -0.7540208101272583,
        6.634061992783456e-15,
        -0.7540208101272583,
        -0.7095873355865479,
        0.3128761947154999,
        -0.7095873355865479,
        -0.3128761947154999,
        -0.7095873355865479,
        0.7095873355865479,
        2.5118773909287538e-14,
        -0.7540208101272583,
        0.7540208101272583,
        0.3128761947154999,
        -0.7095873355865479,
        0.7095873355865479,
        0.7095873355865479,
        -0.3128761947154999,
        0.7095873355865479,
        0.7540208101272583,
        1.9901176315077242e-14,
        0.7540208101272583,
        0.7095873355865479,
        0.3128761947154999,
        0.7095873355865479,
        -0.3128761947154999,
        0.7095873355865479,
        0.7095873355865479,
        1.2922842016047011e-14,
        0.7540208101272583,
        0.7540208101272583,
        0.3128761947154999,
        0.7095873355865479,
        0.7095873355865479,
        -0.7095873355865479,
        -0.3128761947154999,
        0.7095873355865479,
        -0.7540208101272583,
        1.697738459961306e-14,
        0.7540208101272583,
        -0.7095873355865479,
        0.3128761947154999,
        0.7095873355865479,
        -0.7095873355865479,
        -0.7095873355865479,
        -0.3128761947154999,
        -0.7540208101272583,
        -0.7540208101272583,
        1.5042995976210626e-15,
        -0.7095873355865479,
        -0.7095873355865479,
        0.3128761947154999,
        0.7095873355865479,
        -0.7095873355865479,
        -0.3128761947154999,
        0.7540208101272583,
        -0.7540208101272583,
        4.377612808110099e-15,
        0.7095873355865479,
        -0.7095873355865479,
        0.3128761947154999,
        0.7095873355865479,
        0.7095873355865479,
        -0.3128761947154999,
        0.7540208101272583,
        0.7540208101272583,
        1.3068224210014898e-14,
        0.7095873355865479,
        0.7095873355865479,
        0.3128761947154999,
        -0.7095873355865479,
        0.7095873355865479,
        -0.3128761947154999,
        -0.7540208101272583,
        0.7540208101272583,
        8.529075442599825e-15,
        -0.7095873355865479,
        0.7095873355865479,
        0.3128761947154999,
        -1,
        -0.41336411237716675,
        -0.41336411237716675,
        -1.1200461387634277,
        1.356080266796348e-14,
        -0.45730409026145935,
        -1,
        0.41336411237716675,
        -0.41336411237716675,
        -1.1200461387634277,
        -0.45730409026145935,
        4.0291368467526914e-14,
        -1.279833197593689,
        6.071910959052837e-14,
        -1.1453314391460159e-14,
        -1.1200461387634277,
        0.45730409026145935,
        2.2303634498099453e-14,
        -1,
        -0.41336411237716675,
        0.41336411237716675,
        -1.1200461387634277,
        -1.0080084248580753e-14,
        0.45730409026145935,
        -1,
        0.41336411237716675,
        0.41336411237716675,
        1,
        -0.41336411237716675,
        -0.41336411237716675,
        1.1200461387634277,
        -6.3338459877057465e-15,
        -0.45730409026145935,
        1,
        0.41336411237716675,
        -0.41336411237716675,
        1.1200461387634277,
        -0.45730409026145935,
        3.954791409719466e-14,
        1.279833197593689,
        3.969379688763437e-14,
        -9.989605713814353e-14,
        1.1200461387634277,
        0.45730409026145935,
        3.9856884611298715e-14,
        1,
        -0.41336411237716675,
        0.41336411237716675,
        1.1200461387634277,
        -4.3255533161389026e-15,
        0.45730409026145935,
        1,
        0.41336411237716675,
        0.41336411237716675,
        -0.41336411237716675,
        -1,
        -0.41336411237716675,
        -6.174534336920263e-15,
        -1.1200461387634277,
        -0.45730409026145935,
        0.41336411237716675,
        -1,
        -0.41336411237716675,
        -0.45730409026145935,
        -1.1200461387634277,
        2.9226451716667795e-14,
        6.681789588855805e-14,
        -1.279833197593689,
        -9.117169570845539e-15,
        0.45730409026145935,
        -1.1200461387634277,
        2.8000232612113846e-14,
        -0.41336411237716675,
        -1,
        0.41336411237716675,
        6.005567357569028e-15,
        -1.1200461387634277,
        0.45730409026145935,
        0.41336411237716675,
        -1,
        0.41336411237716675,
        -0.41336411237716675,
        1,
        -0.41336411237716675,
        -1.0504915317342042e-14,
        1.1200461387634277,
        -0.45730409026145935,
        0.41336411237716675,
        1,
        -0.41336411237716675,
        -0.45730409026145935,
        1.1200461387634277,
        4.402795267038559e-14,
        8.600322771925174e-15,
        1.279833197593689,
        -1.5215955868875218e-14,
        0.45730409026145935,
        1.1200461387634277,
        2.7498160608892437e-14,
        -0.41336411237716675,
        1,
        0.41336411237716675,
        1.977395704859733e-14,
        1.1200461387634277,
        0.45730409026145935,
        0.41336411237716675,
        1,
        0.41336411237716675,
        -0.41336411237716675,
        -0.41336411237716675,
        -1,
        -7.815927064087382e-15,
        -0.45730409026145935,
        -1.1200461387634277,
        0.41336411237716675,
        -0.41336411237716675,
        -1,
        -0.45730409026145935,
        2.13236258484557e-14,
        -1.1200461387634277,
        1.0651168938806063e-13,
        3.849471333057866e-14,
        -1.279833197593689,
        0.45730409026145935,
        -1.86732580624311e-14,
        -1.1200461387634277,
        -0.41336411237716675,
        0.41336411237716675,
        -1,
        -8.515932879390228e-15,
        0.45730409026145935,
        -1.1200461387634277,
        0.41336411237716675,
        0.41336411237716675,
        -1,
        -0.41336411237716675,
        -0.41336411237716675,
        1,
        -5.252457658671021e-15,
        -0.45730409026145935,
        1.1200461387634277,
        0.41336411237716675,
        -0.41336411237716675,
        1,
        -0.45730409026145935,
        -2.3790543238763964e-14,
        1.1200461387634277,
        5.1932718244046897e-14,
        6.483320282545113e-14,
        1.279833197593689,
        0.45730409026145935,
        -3.336855463709036e-14,
        1.1200461387634277,
        -0.41336411237716675,
        0.41336411237716675,
        1,
        8.96007460528302e-15,
        0.45730409026145935,
        1.1200461387634277,
        0.41336411237716675,
        0.41336411237716675,
        1,
        -0.6340351700782776,
        -0.6340351700782776,
        -0.6340351700782776,
        7.311288212222614e-14,
        -0.7453672289848328,
        -0.7453672289848328,
        0.6340351700782776,
        -0.6340351700782776,
        -0.6340351700782776,
        -0.7453672289848328,
        -4.57577235447198e-15,
        -0.7453672289848328,
        -1.792698205594495e-13,
        1.457957425577902e-13,
        -0.8993141055107117,
        0.7453672289848328,
        1.4125209519839227e-13,
        -0.7453672289848328,
        -0.6340351700782776,
        0.6340351700782776,
        -0.6340351700782776,
        1.8541824976445187e-13,
        0.7453672289848328,
        -0.7453672289848328,
        0.6340351700782776,
        0.6340351700782776,
        -0.6340351700782776,
        -0.7453672289848328,
        -0.7453672289848328,
        -3.460676569529836e-13,
        2.0284624403354296e-13,
        -0.8993141055107117,
        5.85852004929277e-13,
        0.7453672289848328,
        -0.7453672289848328,
        -3.0697463342978237e-13,
        -0.8993141055107117,
        2.124102658357125e-14,
        5.956384474190002e-13,
        -1.2616546262583794e-13,
        -1.5974175378730832e-13,
        -7.702197712088077e-13,
        0.8993141055107117,
        1.6014177695511042e-14,
        5.324714193871705e-13,
        -0.7453672289848328,
        0.7453672289848328,
        -3.1751879771280134e-13,
        4.6263177750499596e-14,
        0.8993141055107117,
        3.5275675067635015e-13,
        0.7453672289848328,
        0.7453672289848328,
        -2.880747127387978e-13,
        -0.6340351700782776,
        -0.6340351700782776,
        0.6340351700782776,
        7.679339613210837e-14,
        -0.7453672289848328,
        0.7453672289848328,
        0.6340351700782776,
        -0.6340351700782776,
        0.6340351700782776,
        -0.7453672289848328,
        4.794613396087201e-14,
        0.7453672289848328,
        1.2455471258608717e-14,
        -1.4234823714730227e-13,
        0.8993141055107117,
        0.7453672289848328,
        6.04797719609522e-14,
        0.7453672289848328,
        -0.6340351700782776,
        0.6340351700782776,
        0.6340351700782776,
        1.2255111205353164e-13,
        0.7453672289848328,
        0.7453672289848328,
        0.6340351700782776,
        0.6340351700782776,
        0.6340351700782776,
    ]
).reshape(-1, 3)

connectivity = arange(125)[None]


def vtk_sphere(radius):
    """Create vtk sphere approximation using Bezier Hexahedron cell

    Args:
        radius (float): radius of sphere

    Returns:
        points:     coordinates of points used to approximate sphere
        cell:       connectivity of Bezier Hexahedron cell
        point_data: additional data needed for Bezier Hexahedron cell
    """
    cell = ("VTB_BEZIER_HEXAHEDRON", connectivity)
    point_data = {"RationalWeights": weights}
    return radius * points, cell, point_data
