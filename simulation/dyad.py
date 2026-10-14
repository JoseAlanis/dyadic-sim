from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agents.agent_factory import build_agent
from agents.base_agent import BaseAgent
from priors.patient_prior import PatientPrior, build_patient_prior
from priors.therapist_prior import TherapistPrior, build_therapist_prior
from simulation.opening import opening_utterance
from simulation.session import Session, new_session, resume_session
from simulation.turn_manager import build_patient_context, build_therapist_context
from simulation.utterance import clean_utterance

console = Console()


class Dyad:
    """
    Manages a complete dyadic simulation (agents, priors, session).

    Each agent's context on every turn is its prior plus the transcript so far;
    nothing else is carried between turns. See simulation/turn_manager.py.
    """

    def __init__(
        self,
        therapist_model: str,
        patient_model: str,
        case_name: str,
        orientation: str = "psychodynamic",
        session_id: str | None = None,
        patient_id: str | None = None,
        patient_prior: PatientPrior | None = None,
    ):
        # Build priors
        self.therapist_prior: TherapistPrior = build_therapist_prior(orientation)
        self.patient_prior: PatientPrior = patient_prior or build_patient_prior(case_name)

        # Build agents
        self.therapist_agent: BaseAgent = build_agent(therapist_model, "therapist")
        self.patient_agent: BaseAgent = build_agent(patient_model, "patient")

        # Session (new or resumed)
        if session_id:
            self.session = resume_session(session_id)

            metadata_changed = False
            if not getattr(self.session, "patient_symptoms", ""):
                self.session.patient_symptoms = self.patient_prior.symptoms
                metadata_changed = True
            if not getattr(self.session, "patient_symptom_levels", {}):
                self.session.patient_symptom_levels = self.patient_prior.symptom_levels
                metadata_changed = True
            if not getattr(self.session, "initial_patient_prompt", None):
                self.session.initial_patient_prompt = opening_utterance()
                metadata_changed = True
            if patient_id is not None and self.session.patient_id != patient_id:
                self.session.patient_id = patient_id
                metadata_changed = True
            if metadata_changed:
                self.session.save_metadata()
        else:
            self.session = new_session(
                therapist_model=therapist_model,
                patient_model=patient_model,
                case_name=case_name,
                orientation=orientation,
                patient_symptoms=self.patient_prior.symptoms,
                patient_symptom_levels=self.patient_prior.symptom_levels,
                patient_id=patient_id,
                conversation_language="en",
                initial_patient_prompt=opening_utterance(),
            )

    def run(self, n_turns: int = 10, max_tokens: int = 300) -> Session:
        """
        Run the dyadic exchange for n_turns turns.
        """
        console.rule(f"[bold]Session: {self.session.session_id}[/bold]")
        console.print(
            f"Therapist: [cyan]{self.therapist_agent.model}[/cyan]  |  "
            f"Patient: [magenta]{self.patient_agent.model}[/magenta]  |  "
            f"Case: [yellow]{self.patient_prior.case_name}[/yellow]  |  "
            f"Orientation: [green]{self.therapist_prior.orientation}[/green]"
        )
        console.rule()

        opening_prompt = opening_utterance()

        start_turn = self.session.turn_count + 1

        for turn_num in range(start_turn, start_turn + n_turns):
            console.print(f"\n[dim]-- Turn {turn_num} --[/dim]")

            history = self.session.get_history()

            # --- Patient turn ---

            # First turn: the patient answers the opening; afterwards, the
            # therapist's latest utterance.
            latest_therapist = history[-1][1] if history else opening_prompt

            patient_system, patient_messages = build_patient_context(
                prior=self.patient_prior,
                history=history,
                latest_therapist_turn=latest_therapist,
            )

            patient_response = self.patient_agent.complete(
                system_prompt=patient_system,
                messages=patient_messages,
                max_tokens=max_tokens,
            )
            patient_text_raw = patient_response.content.strip()
            patient_text = clean_utterance(patient_text_raw)

            symptom_discussion_started = self.patient_prior.check_symptom_discussion(
                patient_text=patient_text,
                current_turn=turn_num,
            )
            if symptom_discussion_started:
                console.print(
                    "[bold magenta]Patient began talking explicitly about symptoms[/bold magenta]"
                )
                self.patient_prior.symptom_discussion.started = False

            _print_turn("Patient", patient_text, "magenta")

            # --- Therapist turn ---

            therapist_system, therapist_messages = build_therapist_context(
                prior=self.therapist_prior,
                history=history,
                latest_patient_turn=patient_text,
            )

            therapist_response = self.therapist_agent.complete(
                system_prompt=therapist_system,
                messages=therapist_messages,
                max_tokens=max_tokens,
            )
            therapist_text_raw = therapist_response.content.strip()
            therapist_text = clean_utterance(therapist_text_raw)

            _print_turn("Therapist", therapist_text, "cyan")

            # --- Record turn ---

            self.session.append_turn(
                therapist_text=therapist_text,
                patient_text=patient_text,
                therapist_tokens=therapist_response.output_tokens,
                patient_tokens=patient_response.output_tokens,
                symptom_discussion_started=symptom_discussion_started,
                therapist_text_raw=(
                    therapist_text_raw if therapist_text_raw != therapist_text else None
                ),
                patient_text_raw=(
                    patient_text_raw if patient_text_raw != patient_text else None
                ),
            )

        console.rule("[bold]Session complete[/bold]")
        self.session.save_metadata()
        return self.session


def _print_turn(label: str, text: str, colour: str) -> None:
    """Render a turn in the terminal with a coloured panel."""
    console.print(
        Panel(
            Text(text),
            title=f"[bold {colour}]{label}[/bold {colour}]",
            border_style=colour,
            padding=(0, 1),
        )
    )
