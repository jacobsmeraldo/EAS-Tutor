"""Streamlit UI for an aromatic EAS and synthesis playground."""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from eas_generator import BENZENE_SMILES, EASGenerator, ReactionOutcome
from tutor_logic import get_single_step_tutor_response, get_synthesis_tutor_response


APP_TITLE = "Organic Chemistry Aromatic Synthesis Playground"
MODE_SINGLE_STEP = "Mode 1: Single-Step Practice"
MODE_MULTI_STEP = "Mode 2: Multi-Step Synthesis"


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    engine = get_engine()
    mode = render_sidebar()
    initialize_session_state(engine)

    if mode == MODE_SINGLE_STEP:
        render_single_step_mode(engine)
    else:
        render_multi_step_mode(engine)


@st.cache_resource
def get_engine() -> EASGenerator:
    """Create one reusable reaction engine per Streamlit session."""

    return EASGenerator()


def render_sidebar() -> str:
    """Render app-level controls."""

    with st.sidebar:
        st.header("Workspace")
        mode = st.selectbox(
            "Practice Mode",
            [MODE_SINGLE_STEP, MODE_MULTI_STEP],
            key="practice_mode",
        )
        st.caption(
            "Single-step mode focuses on regioselectivity. Multi-step mode "
            "turns benzene into a target using a reaction flask."
        )
    return mode


def initialize_session_state(engine: EASGenerator) -> None:
    """Initialize all mode-specific session-state values."""

    if "single_problem" not in st.session_state:
        st.session_state.single_problem = None
    if "single_chat_history" not in st.session_state:
        st.session_state.single_chat_history = []

    if "synthesis_challenge" not in st.session_state:
        st.session_state.synthesis_challenge = (
            engine.generate_synthesis_challenge().to_dict()
        )
    if "current_intermediate" not in st.session_state:
        st.session_state.current_intermediate = BENZENE_SMILES
    if "route_history" not in st.session_state:
        st.session_state.route_history = []
    if "synthesis_chat_history" not in st.session_state:
        st.session_state.synthesis_chat_history = []
    if "last_outcome" not in st.session_state:
        st.session_state.last_outcome = None


def render_single_step_mode(engine: EASGenerator) -> None:
    """Render the single-step EAS prediction workspace."""

    st.subheader("Single-Step EAS Practice")

    if st.button("Generate New EAS Problem", type="primary"):
        st.session_state.single_problem = engine.generate_single_step_problem()
        st.session_state.single_chat_history = []

    problem = st.session_state.single_problem
    if problem is None:
        st.info("Generate a single-step problem to begin.")
        return

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**Substrate:** {problem['substrate_name']}")
        render_molecule(problem["substrate_smiles"], "Substrate")
    with right:
        st.markdown(f"**Reagent:** {problem['reagent']}")
        st.markdown("Predict the major product and explain the mechanism.")

    st.divider()
    render_chat_history(st.session_state.single_chat_history)

    student_text = st.chat_input(
        "Explain your predicted product or ask for help...",
        key="single_step_chat_input",
    )
    if student_text:
        st.session_state.single_chat_history.append(
            {"role": "user", "content": student_text}
        )
        with st.chat_message("user"):
            st.markdown(student_text)

        with st.chat_message("assistant"):
            with st.spinner("Tutor is thinking..."):
                tutor_text = get_single_step_tutor_response(
                    st.session_state.single_chat_history,
                    problem,
                )
            st.markdown(tutor_text)

        st.session_state.single_chat_history.append(
            {"role": "assistant", "content": tutor_text}
        )


def render_multi_step_mode(engine: EASGenerator) -> None:
    """Render the target-driven synthesis builder workspace."""

    st.subheader("Multi-Step Target Synthesis")

    controls_left, controls_right = st.columns([1, 1])
    with controls_left:
        if st.button("Generate New Target", type="primary"):
            new_synthesis_challenge(engine)
    with controls_right:
        if st.button("Reset Flask"):
            reset_flask()

    challenge = st.session_state.synthesis_challenge

    target_col, flask_col = st.columns([1, 1])
    with target_col:
        st.markdown(f"**Target Molecule:** {challenge['name']}")
        render_molecule(challenge["target_smiles"], "Target Molecule")
        # st.caption(challenge["learning_goal"])

    with flask_col:
        st.markdown("**Current Reaction Flask**")
        render_molecule(st.session_state.current_intermediate, "Current Intermediate")
        st.code(st.session_state.current_intermediate, language="text")

    st.divider()
    render_reagent_controls(engine)
    render_route_history()
    render_target_status(engine)

    st.divider()
    st.subheader("Synthesis Route Advisor")
    render_chat_history(st.session_state.synthesis_chat_history)
    render_hint_button()
    render_synthesis_chat_input()


def render_reagent_controls(engine: EASGenerator) -> None:
    """Render reagent dropdown and step-application button."""

    reagents = engine.list_reagents()
    reagent_labels = {
        f"{reagent.category}: {reagent.label}": reagent.key for reagent in reagents
    }

    selected_label = st.selectbox(
        "Available reagents",
        list(reagent_labels.keys()),
        key="selected_reagent_label",
    )

    if st.button("Add Reagent to Flask"):
        reagent_key = reagent_labels[selected_label]
        outcome = engine.apply_reagent(
            st.session_state.current_intermediate,
            reagent_key,
        )
        st.session_state.last_outcome = outcome.to_dict()

        if outcome.success and outcome.product_smiles is not None:
            step_number = len(st.session_state.route_history) + 1
            st.session_state.route_history.append(
                {
                    "step_number": step_number,
                    "reagent_key": outcome.reagent_key,
                    "reagent_label": outcome.reagent_label,
                    "substrate_smiles": outcome.substrate_smiles,
                    "product_smiles": outcome.product_smiles,
                    "message": outcome.message,
                }
            )
            st.session_state.current_intermediate = outcome.product_smiles
            st.success(outcome.message)
        else:
            st.warning(outcome.message)

        prompt_synthesis_coach_after_reagent(outcome)
        st.rerun()


def render_route_history() -> None:
    """Display the route history accumulated in the flask."""

    if not st.session_state.route_history:
        st.info("Your flask is still benzene. Add a reagent to begin the route.")
        return

    st.markdown("**Route So Far**")
    for step in st.session_state.route_history:
        st.markdown(
            f"{step['step_number']}. {step['reagent_label']} -> "
            f"`{step['product_smiles']}`"
        )


def render_target_status(engine: EASGenerator) -> None:
    """Show whether the current intermediate matches the target."""

    current = engine.canonicalize_smiles(st.session_state.current_intermediate)
    target = engine.canonicalize_smiles(
        st.session_state.synthesis_challenge["target_smiles"]
    )

    if current == target:
        st.success("Target reached. Now explain why this order worked.")


def render_hint_button() -> None:
    """Render a dedicated Socratic hint button."""

    if st.button("Request Hint"):
        background_message = (
            "Please give me a targeted Socratic hint for the current flask and "
            "target. Do not reveal the exact next reagent."
        )
        st.session_state.synthesis_chat_history.append(
            {"role": "user", "content": background_message, "hidden": True}
        )
        with st.spinner("Coach is preparing a hint..."):
            hint = get_synthesis_tutor_response(
                st.session_state.synthesis_chat_history,
                build_current_route_data(),
            )
        st.session_state.synthesis_chat_history.append(
            {"role": "assistant", "content": hint}
        )
        st.rerun()


def render_synthesis_chat_input() -> None:
    """Handle free-form student chat in multi-step mode."""

    student_text = st.chat_input(
        "Ask about your route, directing effects, or next strategic choice...",
        key="synthesis_chat_input",
    )
    if not student_text:
        return

    st.session_state.synthesis_chat_history.append(
        {"role": "user", "content": student_text}
    )
    with st.chat_message("user"):
        st.markdown(student_text)

    with st.chat_message("assistant"):
        with st.spinner("Coach is thinking..."):
            tutor_text = get_synthesis_tutor_response(
                st.session_state.synthesis_chat_history,
                build_current_route_data(),
            )
        st.markdown(tutor_text)

    st.session_state.synthesis_chat_history.append(
        {"role": "assistant", "content": tutor_text}
    )


def prompt_synthesis_coach_after_reagent(outcome: ReactionOutcome) -> None:
    """Ask Gemini to comment after a reagent is added to the flask."""

    status = "succeeded" if outcome.success else "failed"
    message = (
        f"I added {outcome.reagent_label} to the flask. The step {status}. "
        f"Engine note: {outcome.message}. Coach me Socratically on what this "
        "means for my route."
    )
    st.session_state.synthesis_chat_history.append(
        {"role": "user", "content": message}
    )

    tutor_text = get_synthesis_tutor_response(
        st.session_state.synthesis_chat_history,
        build_current_route_data(),
    )
    st.session_state.synthesis_chat_history.append(
        {"role": "assistant", "content": tutor_text}
    )


def build_current_route_data() -> dict[str, Any]:
    """Build hidden route state for the synthesis coach."""

    challenge = st.session_state.synthesis_challenge
    return {
        "initial_starting_material": {
            "name": "benzene",
            "smiles": BENZENE_SMILES,
        },
        "target_molecule": {
            "name": challenge["name"],
            "smiles": challenge["target_smiles"],
        },
        "recommended_route": challenge.get("recommended_route", []),
        "learning_goal": challenge.get("learning_goal", ""),
        "steps_used_so_far": st.session_state.route_history,
        "current_intermediate": st.session_state.current_intermediate,
        "last_engine_outcome": st.session_state.last_outcome,
    }


def new_synthesis_challenge(engine: EASGenerator) -> None:
    """Reset all synthesis state around a new target."""

    st.session_state.synthesis_challenge = engine.generate_synthesis_challenge().to_dict()
    reset_flask()


def reset_flask() -> None:
    """Clear the multi-step flask while keeping the same target."""

    st.session_state.current_intermediate = BENZENE_SMILES
    st.session_state.route_history = []
    st.session_state.synthesis_chat_history = []
    st.session_state.last_outcome = None


def render_chat_history(chat_history: list[dict[str, str]]) -> None:
    """Render chat messages with Streamlit chat bubbles."""

    for message in chat_history:
        if message.get("hidden"):
            continue
        role = "assistant" if message.get("role") == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(message.get("content", ""))


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert a SMILES string to an RDKit molecule."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return mol


def molecule_image(smiles: str) -> Optional[Any]:
    """Render a SMILES string as a 2D textbook-style molecule image."""

    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(300, 300))


def render_molecule(smiles: str, caption: str) -> None:
    """Render a molecule image, falling back to SMILES text if needed."""

    image = molecule_image(smiles)
    if image is None:
        st.code(smiles, language="text")
        return
    st.image(image, caption=caption)


if __name__ == "__main__":
    main()


