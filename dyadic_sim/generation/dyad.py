import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from dyadic_sim.generation.agents.agent_factory import build_agent
from dyadic_sim.generation.agents.base_agent import BaseAgent
from dyadic_sim.generation.opening import opening_utterance
from dyadic_sim.generation.priors.patient_prior import PatientPrior, build_patient_prior
from dyadic_sim.generation.priors.therapist_prior import (
    TherapistPrior,
    build_therapist_prior,
)
from dyadic_sim.generation.session import (
    Session,
    check_continuable,
    completion_record,
    new_session,
    resume_session,
)
from dyadic_sim.generation.turn_manager import (
    build_patient_context,
    build_therapist_context,
)
from dyadic_sim.generation.utterance import clean_utterance

console = Console()


class Dyad:
    """
    Manages a complete dyadic simulation (agents, priors, session).

    Each agent's context on every turn is its prior plus the transcript so far;
    nothing else is carried between turns. See dyadic_sim/generation/turn_manager.py.
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
        # A saved session is checked before anything is built, written or sent:
        # one from another generator version, or one whose patient prompt the
        # caller cannot rebuild, must not grow (dyadic_sim/generation/session.py,
        # check_continuable).
        resumed = None
        if session_id:
            resumed = resume_session(session_id)
            check_continuable(resumed, prior_supplied=patient_prior is not None)

        # Build priors
        self.therapist_prior: TherapistPrior = build_therapist_prior(orientation)
        self.patient_prior: PatientPrior = patient_prior or build_patient_prior(case_name)

        # Build agents
        self.therapist_agent: BaseAgent = build_agent(therapist_model, "therapist")
        self.patient_agent: BaseAgent = build_agent(patient_model, "patient")

        # Session (new or resumed). A resumed record is taken as saved: its
        # opening, symptoms and version are never back-filled from the current
        # configuration.
        if resumed is not None:
            self.session = resumed
            if patient_id is not None and self.session.patient_id != patient_id:
                self.session.patient_id = patient_id
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
            if patient_prior is not None:
                # Marks the record so a plain resume cannot rebuild it from case_name.
                self.session.patient_prompt_source = "supplied"
                self.session.save_metadata({"patient_prompt_source": "supplied"})

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

        # The opening as saved with the session, not as the code now words it.
        opening = self.session.initial_patient_prompt

        start_turn = self.session.turn_count + 1

        # Diagnostics only record; nothing below changes what is sent. The
        # prompts are stored as rendered before the first completion, and each
        # completion stores the hash of the system prompt it was actually sent.
        self.session.begin_diagnostics(start_turn)
        if opening:
            self.session.record_prompt("opening", opening, start_turn)
        self.session.record_prompt(
            "patient_system", self.patient_prior.build_system_prompt(), start_turn
        )
        self.session.record_prompt(
            "therapist_system", self.therapist_prior.build_system_prompt(), start_turn
        )

        for turn_num in range(start_turn, start_turn + n_turns):
            console.print(f"\n[dim]-- Turn {turn_num} --[/dim]")

            history = self.session.get_history()

            # --- Patient turn ---

            # The patient answers the opening on the first turn and the
            # therapist's latest line afterwards; both are the end of `history`.
            patient_system, patient_messages = build_patient_context(
                prior=self.patient_prior,
                opening=opening,
                history=history,
            )

            patient_prompt_sha = self.session.record_prompt(
                "patient_system", patient_system, turn_num
            )
            started = time.perf_counter()
            patient_response = self.patient_agent.complete(
                system_prompt=patient_system,
                messages=patient_messages,
                max_tokens=max_tokens,
            )
            patient_diagnostics = completion_record(
                "patient", turn_num, patient_prompt_sha, patient_messages,
                patient_response, time.perf_counter() - started,
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
                opening=opening,
                history=history,
                patient_turn=patient_text,
            )

            therapist_prompt_sha = self.session.record_prompt(
                "therapist_system", therapist_system, turn_num
            )
            started = time.perf_counter()
            therapist_response = self.therapist_agent.complete(
                system_prompt=therapist_system,
                messages=therapist_messages,
                max_tokens=max_tokens,
            )
            therapist_diagnostics = completion_record(
                "therapist", turn_num, therapist_prompt_sha, therapist_messages,
                therapist_response, time.perf_counter() - started,
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
                patient_diagnostics=patient_diagnostics,
                therapist_diagnostics=therapist_diagnostics,
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
