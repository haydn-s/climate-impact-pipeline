"""Streamlit front end for the model prediction API."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_TIMEOUT_SECONDS = 60


def _configured_api_url() -> str | None:
    api_url = os.environ.get("API_URL")
    if not api_url:
        try:
            api_url = st.secrets["API_URL"]
        except (FileNotFoundError, KeyError):
            return None

    api_url = str(api_url).strip().rstrip("/")
    return api_url or None


@st.cache_data(ttl=300)
def _load_metadata(api_url: str) -> dict[str, Any]:
    response = requests.get(
        f"{api_url}/metadata",
        timeout=API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    metadata = response.json()
    if not isinstance(metadata, dict) or not isinstance(metadata.get("features"), list):
        raise ValueError("The prediction API returned invalid model metadata.")
    return metadata


def _feature_label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def main() -> None:
    st.set_page_config(page_title="Model predictions", page_icon="🌦️")
    st.title("Model predictions")

    api_url = _configured_api_url()
    if api_url is None:
        st.error("Set API_URL in the environment or Streamlit Secrets to connect to the API.")
        st.stop()

    try:
        metadata = _load_metadata(api_url)
    except requests.RequestException as exc:
        st.error(f"Could not load model details from the prediction API: {exc}")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    target = str(metadata.get("target", "target")).replace("_", " ")
    st.write(f"Enter the model inputs to predict **{target}**.")

    features = metadata["features"]
    if not features:
        st.error("The prediction API did not provide any model features.")
        st.stop()

    payload: dict[str, Any] = {}
    with st.form("prediction_form"):
        for feature in features:
            name = feature["name"]
            label = _feature_label(name)
            if feature["kind"] == "categorical":
                categories = feature.get("categories", [])
                if not categories:
                    st.error(f"No categories are available for {label}.")
                    st.stop()
                example = feature.get("example")
                index = categories.index(example) if example in categories else 0
                payload[name] = st.selectbox(label, categories, index=index)
            elif feature["kind"] == "numeric":
                example = float(feature["example"])
                minimum = feature.get("min")
                maximum = feature.get("max")
                help_text = (
                    f"Observed in training from {minimum:g} to {maximum:g}."
                    if minimum is not None and maximum is not None
                    else None
                )
                payload[name] = st.number_input(
                    label,
                    value=example,
                    step=0.1,
                    help=help_text,
                )
            else:
                st.error(f"Unsupported feature type for {label}.")
                st.stop()

        submitted = st.form_submit_button("Get prediction", type="primary")

    if not submitted:
        return

    try:
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or "prediction" not in result:
            raise ValueError("The prediction API returned an invalid prediction response.")
    except requests.RequestException as exc:
        st.error(f"Could not get a prediction from the API: {exc}")
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    st.success(f"Predicted {target}: **{result['prediction']}**")
    probability = result.get("probability")
    if probability is not None:
        st.metric("Confidence", f"{probability:.1%}")
    st.caption(
        "This is an estimate from the current trained model, not a guarantee or official advisory."
    )

    with st.expander("Model details"):
        st.write(f"Model: {metadata.get('model_type', 'unknown')}")
        st.write(f"Task: {metadata.get('task', 'unknown')}")
        model_version = result.get("model_version", metadata.get("model_version", "unknown"))
        st.write(f"Model version: {model_version}")
        st.json(metadata.get("metrics", {}))


if __name__ == "__main__":
    main()
