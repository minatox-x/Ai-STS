from app.router import route, RouteDecision


def test_local_when_gemini_disabled():
    r = route("what's the weather today?", gemini_enabled=False)
    assert r.decision == RouteDecision.LOCAL


def test_current_info_routes_to_gemini():
    r = route("what's the weather today?", gemini_enabled=True)
    assert r.decision == RouteDecision.GEMINI_CURRENT_INFO


def test_simple_question_stays_local():
    r = route("explain python lists to me", gemini_enabled=True)
    assert r.decision == RouteDecision.LOCAL


def test_web_pattern_routes_to_gemini():
    r = route("search for the best pizza place nearby", gemini_enabled=True)
    assert r.decision == RouteDecision.GEMINI_WEB


def test_math_question_stays_local():
    r = route("what's 20% of 500?", gemini_enabled=True)
    assert r.decision == RouteDecision.LOCAL
