import re
from dataclasses import dataclass, field

from dyadic_sim.generation.priors.loader import load_patient_case, load_symptom_facts
from dyadic_sim.generation.utterance import SPEECH_ONLY_INSTRUCTION
from dyadic_sim.scoring.config import PHQ_SYMPTOM_LABELS


@dataclass
class SymptomDiscussion:
    """
    Tracks whether the patient has begun talking explicitly about symptoms.
    """
    started: bool = False
    start_turn: int | None = None


@dataclass
class PatientPrior:
    """
    Complete patient prior, as rendered into the system prompt by
    build_system_prompt().
    """
    case_name: str

    # Visible to the patient agent from the start. Only presenting_complaint is
    # required; a field left empty is left out of the prompt.
    presenting_complaint: str = ""
    everyday_context: str = ""
    theory_of_cure: str = ""
    relational_pattern: str = ""
    transference_expectation: str = ""
    resistance_structure: str = ""
    symptoms: str = ""
    symptom_levels: dict[str, str] = field(default_factory=dict)
    # Graph-first experiments supply a derived intake profile instead of the
    # source case and symptom block. The two inputs must never be mixed.
    profile_text: str = ""

    # Runtime tracking
    symptom_discussion: SymptomDiscussion = field(default_factory=SymptomDiscussion)

    def build_system_prompt(self) -> str:
        """
        Render the system prompt for a patient turn.

        Returns:
            System prompt string passed to the LLM on each turn.
        """
        if self.profile_text.strip():
            return (
                "## Who You Are in This Encounter\n"
                "You are the person described in the intake profile below, speaking "
                "in a therapy session. Speak from inside your experience; do not "
                "recite the profile or use clinical terminology.\n\n"
                "## Your Intake Profile\n" + self.profile_text.strip() + "\n\n"
                "## How to Respond\n"
                "Speak naturally in one to five sentences. Stay consistent with "
                "your profile, but reveal details only when they fit the conversation.\n\n"
                + SPEECH_ONLY_INSTRUCTION
            )

        sections = []

        sections.append(
            "## Who You Are in This Encounter\n"
            "You are a person at a first appointment with a therapist. You are "
            "not performing being a patient; you are yourself. Below, in your "
            "own words, is why you came and how things are for you. Speak from "
            "inside your life, not from above it."
        )

        for heading, text in (
            ("What Brings You Here", self.presenting_complaint),
            ("Your Everyday Life", self.everyday_context),
            ("What You Believe Would Help", self.theory_of_cure),
            ("How You Tend to Relate", self.relational_pattern),
            ("What You Expect From This Person", self.transference_expectation),
            ("What You Do When Things Get Close", self.resistance_structure),
        ):
            if text.strip():
                sections.append(f"## {heading}\n" + text.strip())

        if self.symptoms.strip():
            sections.append(
                "## Your Health Lately\n"
                "These are facts about your health. They are not necessarily why "
                "you came. Mention them only as described, in your own words, "
                "never as a list.\n\n"
                + self.symptoms.strip()
                + "\n\nDo not add symptoms, causes or details that this prompt "
                "does not describe. If you are asked about a detail it doesn't "
                "cover, say you don't know or keep it brief."
            )

        sections.append(
            "## How to Respond\n"
            "Talk the way people talk at a first appointment: everyday words, "
            "plain and direct. Answer what you were asked, usually in one to four "
            "sentences. Treat what the therapist says about you as a question, "
            "not a fact: if it doesn't fit what is described here, say so plainly. "
            "Asked how life has been, talk about it naturally, "
            "including recent activities, interests or plans. Figures of speech "
            "and quoting your own thoughts are rare in ordinary talk; use them "
            "seldom. Don't ask rhetorical questions, and don't repeat what you "
            "have already said unless you are confirming or clarifying something. "
            "It is fine to say you don't know.\n\n"
            + SPEECH_ONLY_INSTRUCTION
        )

        return "\n\n---\n\n".join(sections)

    def check_symptom_discussion(self, patient_text: str, current_turn: int) -> bool:
        """
        Mark symptom discussion as started once the patient explicitly talks
        about symptom experience in their own turn.

        This version is more general than simple keyword matching.
        It detects symptom discussion by looking for different symptom categories,
        natural phrases, and common ways patients describe distress.

        Returns True only on the first turn where this is detected.
        """
        if self.symptom_discussion.started:
            return False

        if not patient_text:
            return False

        text = patient_text.lower().strip()

        # Normalize common apostrophe variants
        text = (
            text.replace("’", "'")
            .replace("‘", "'")
            .replace("`", "'")
        )

        if not text:
            return False

        symptom_patterns = {
            "depressed_mood": [
                r"\bdepressed\b",
                r"\bdepression\b",
                r"\bdown\b",
                r"\bsad\b",
                r"\blow mood\b",
                r"\bempty\b",
                r"\bnumb\b",
                r"\bhopeless\b",
                r"\bmiserable\b",
                r"\bi feel low\b",
                r"\bi feel awful\b",
            ],

            "lack_of_pleasure": [
                r"\bno interest\b",
                r"\blost interest\b",
                r"\bdon't enjoy\b",
                r"\bdo not enjoy\b",
                r"\bcan't enjoy\b",
                r"\bcannot enjoy\b",
                r"\bnothing feels good\b",
                r"\bnothing is fun\b",
                r"\bno pleasure\b",
                r"\bhard to enjoy\b",
                r"\bi don't care about anything\b",
            ],

            "sleep_problems": [
                r"\bcan't sleep\b",
                r"\bcannot sleep\b",
                r"\bcan't fall asleep\b",
                r"\bwake up at night\b",
                r"\bwaking up\b",
                r"\binsomnia\b",
                r"\bsleep badly\b",
                r"\bpoor sleep\b",
                r"\bsleeping too much\b",
                r"\boversleeping\b",
                r"\bnightmares?\b",
                r"\btired because i didn't sleep\b",
            ],

            "low_energy": [
                r"\bexhausted\b",
                r"\btired\b",
                r"\bfatigued\b",
                r"\bno energy\b",
                r"\blow energy\b",
                r"\bdrained\b",
                r"\bworn out\b",
                r"\bi can't get out of bed\b",
                r"\beverything feels like effort\b",
                r"\beven small things feel hard\b",
            ],

            "appetite_changes": [
                r"\bappetite\b",
                r"\bnot hungry\b",
                r"\bcan't eat\b",
                r"\bcannot eat\b",
                r"\beating too much\b",
                r"\bovereating\b",
                r"\blost weight\b",
                r"\bgained weight\b",
                r"\bfood doesn't taste\b",
            ],

            "feelings_of_failure_or_guilt": [
                r"\bguilty\b",
                r"\bashamed\b",
                r"\bshame\b",
                r"\bfailure\b",
                r"\bworthless\b",
                r"\buseless\b",
                r"\bmy fault\b",
                r"\bnot good enough\b",
                r"\bi let .* down\b",
                r"\bi hate myself\b",
            ],

            "concentration_problems": [
                r"\bcan't focus\b",
                r"\bcannot focus\b",
                r"\bhard to focus\b",
                r"\bcan't concentrate\b",
                r"\bcannot concentrate\b",
                r"\bconcentration\b",
                r"\bdistracted\b",
                r"\bcan't think clearly\b",
                r"\bbrain fog\b",
                r"\bfoggy\b",
            ],

            "psychomotor_changes": [
                r"\brestless\b",
                r"\bagitated\b",
                r"\bcan't sit still\b",
                r"\bcannot sit still\b",
                r"\bslowed down\b",
                r"\bmoving slowly\b",
                r"\btalking slowly\b",
                r"\beverything feels slow\b",
                r"\bmy body feels heavy\b",
            ],

            "fear_or_anxiety": [
                r"\banxious\b",
                r"\banxiety\b",
                r"\bafraid\b",
                r"\bscared\b",
                r"\bterrified\b",
                r"\bpanic\b",
                r"\bpanicking\b",
                r"\bheart races\b",
                r"\bi freeze\b",
                r"\bi avoid\b",
                r"\bi can't go\b",
                r"\bi cannot go\b",
            ],

            "thoughts_of_death_or_self_harm": [
                r"\bsuicidal\b",
                r"\bself[- ]?harm\b",
                r"\bhurt myself\b",
                r"\bkill myself\b",
                r"\bbetter off dead\b",
                r"\bdon't want to be here\b",
                r"\bdo not want to be here\b",
                r"\bi wish i wouldn't wake up\b",
                r"\bi wish i were dead\b",
                r"\bend it\b",
            ],
        }

        matched_categories = []

        for category, patterns in symptom_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    matched_categories.append(category)
                    break

        if matched_categories:
            self.symptom_discussion.started = True
            self.symptom_discussion.start_turn = current_turn
            return True

        # More general backup detection:
        # This catches natural symptom talk that may not use exact clinical words.
        experience_patterns = [
            r"\bi feel\b",
            r"\bi've been feeling\b",
            r"\bi have been feeling\b",
            r"\bi keep feeling\b",
            r"\bit feels like\b",
            r"\bi can't\b",
            r"\bi cannot\b",
            r"\bi don't feel\b",
            r"\bi do not feel\b",
            r"\bi struggle\b",
            r"\bi'm struggling\b",
        ]

        distress_patterns = [
            r"\bhard\b",
            r"\bheavy\b",
            r"\bunbearable\b",
            r"\boverwhelming\b",
            r"\btoo much\b",
            r"\bpointless\b",
            r"\bempty\b",
            r"\bscary\b",
            r"\bafraid\b",
            r"\bexhausting\b",
            r"\bIrrational\b".lower(),
        ]

        has_experience_language = any(
            re.search(pattern, text) for pattern in experience_patterns
        )

        has_distress_language = any(
            re.search(pattern, text) for pattern in distress_patterns
        )

        if has_experience_language and has_distress_language:
            self.symptom_discussion.started = True
            self.symptom_discussion.start_turn = current_turn
            return True

        return False


def _format_symptoms(symptoms_raw, absent=()) -> str:
    """The lines of the "Your Health Lately" section.

    `symptoms_raw` maps a symptom key to its PHQ-9 frequency anchor; "not at
    all" means absent, like every key in `absent`. A present symptom gets its
    facts and disclosure rule from symptom_facts.yaml; an absent one, its absent
    line. Text that is not a mapping is used as written.
    """
    if symptoms_raw and not isinstance(symptoms_raw, dict):
        return str(symptoms_raw).strip()
    levels = dict(symptoms_raw or {})
    for key in absent:
        levels.setdefault(key, "not at all")
    if not levels:
        return ""

    facts = load_symptom_facts()
    default = facts["default"]
    lines = []
    for key, value in levels.items():
        frequency = str(value).strip().lower()
        own = facts["symptoms"].get(key, {})
        label = PHQ_SYMPTOM_LABELS.get(key, key.replace("_", " ").strip().capitalize())
        if frequency == "not at all":
            template = own.get("absent", default["absent"])
            lines.append("- " + template.format(label=label))
            continue
        how_often = own.get("how_often", default["how_often"]).get(frequency)
        if how_often is None:
            raise ValueError(f"Unknown frequency {value!r} for symptom {key!r}")
        text = own.get("present", default["present"]).format(label=label, how_often=how_often)
        topic = own.get("topic", label.lower())
        text += " " + facts["disclosure"]["when_asked"].format(topic=topic)
        lines.append("- " + " ".join(text.split()))
    return "\n".join(lines)


def build_patient_prior(case_name: str) -> PatientPrior:
    """
    Load a patient case file and return a PatientPrior instance.

    Args:
        case_name: filename without .yaml, e.g. "only_love_can_save_me"

    Returns:
        Fully populated PatientPrior instance
    """
    raw = load_patient_case(case_name)

    symptoms_raw = raw.get("symptoms", {})
    symptom_levels = (
        {str(key): str(value) for key, value in symptoms_raw.items()}
        if isinstance(symptoms_raw, dict)
        else {}
    )

    return PatientPrior(
        case_name=case_name,
        presenting_complaint=raw.get("presenting_complaint", ""),
        everyday_context=raw.get("everyday_context", ""),
        theory_of_cure=raw.get("theory_of_cure", ""),
        relational_pattern=raw.get("relational_pattern", ""),
        transference_expectation=raw.get("transference_expectation", ""),
        resistance_structure=raw.get("resistance_structure", ""),
        symptoms=_format_symptoms(symptoms_raw, raw.get("symptoms_absent", ())),
        symptom_levels=symptom_levels,
    )
