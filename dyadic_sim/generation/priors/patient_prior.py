import re
from dataclasses import dataclass, field

from dyadic_sim.generation.priors.loader import load_patient_case
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

    # Visible to the patient agent from the start
    presenting_complaint: str = ""
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
            "You are a person in a therapy session. You are not performing "
            "being a patient, you are one. You have come because something "
            "is wrong and you don't fully understand it. Speak from inside "
            "your experience, not from above it."
        )

        sections.append(
            "## What Brings You Here\n" + self.presenting_complaint.strip()
        )

        sections.append(
            "## What You Believe Would Help\n" + self.theory_of_cure.strip()
        )

        sections.append(
            "## How You Tend to Relate\n" + self.relational_pattern.strip()
        )

        sections.append(
            "## What You Expect From This Person\n"
            + self.transference_expectation.strip()
        )

        sections.append(
            "## What You Do When Things Get Close\n"
            + self.resistance_structure.strip()
        )

        if self.symptoms.strip():
            sections.append(
                "## How You've Been Lately\n"
                "Over the past two weeks, this is how much problems in the "
                "following areas have bothered or impaired you. This is simply "
                "how things have been for you. Some areas may have been fine, "
                "others harder.\n\n"
                + self.symptoms.strip()
            )

        sections.append(
            "## How to Respond\n"
            "Speak as this person speaks, in their register, their rhythm. "
            "You are not trying to be a good patient. You are trying to get "
            "something, avoid something, or understand something. "
            "One to five sentences. Raw, not polished.\n\n"
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


def _format_symptoms(symptoms_raw) -> str:
    if not symptoms_raw:
        return ""

    frequency_meanings = {
        "not at all": (
            "over the last two weeks, you have not been bothered by this symptom; "
        ),
        "several days": (
            "over the last two weeks, you have been bothered by this symptom on several days; "
        ),
        "more than half the days": (
            "over the last two weeks, you have been bothered by this symptom more than half the days; "
        ),
        "nearly every day": (
            "over the last two weeks, you have been bothered by this symptom nearly every day; "
        ),
    }

    # Imported, not redeclared: these strings go into the prompt and are the
    # same ones the analysis compares reference wordings against.
    symptom_labels = PHQ_SYMPTOM_LABELS

    if isinstance(symptoms_raw, dict):
        lines = []
        for key, value in symptoms_raw.items():
            frequency = str(value).strip().lower()

            label = symptom_labels.get(
                key,
                key.replace("_", " ").strip().capitalize()
            )

            meaning = frequency_meanings.get(
                frequency,
                "frequency level not recognized; use common sense to decide how much it should affect you"
            )

            lines.append(f"- {label}: {frequency}; {meaning}")

        return "\n".join(lines)
    return str(symptoms_raw).strip()

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
        theory_of_cure=raw.get("theory_of_cure", ""),
        relational_pattern=raw.get("relational_pattern", ""),
        transference_expectation=raw.get("transference_expectation", ""),
        resistance_structure=raw.get("resistance_structure", ""),
        symptoms=_format_symptoms(symptoms_raw),
        symptom_levels=symptom_levels,
    )
