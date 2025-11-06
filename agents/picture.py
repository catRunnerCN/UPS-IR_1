import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Dict

try:
    from volcenginesdkarkruntime import Ark
except ImportError:  # pragma: no cover
    Ark = None  # type: ignore


def to_base64_if_local(path: str, base_dir: str) -> str:
    """
    Convert a local image path into a base64 data URI so Ark can consume it.
    Leaves http/https/data URLs untouched.
    """
    if path.startswith(("http://", "https://", "data:")):
        return path

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate
    candidate = candidate.resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"Image file not found: {candidate}")

    mime = mimetypes.guess_type(candidate)[0] or "image/png"
    encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def describe_images_in_markdown(
    input_md_path: str,
    output_md_path: str,
    model: str = "doubao-1-5-vision-pro-32k-250115",
):
    """
    Retrieve all image URLs from the specified Markdown file,
    call a large model to obtain image descriptions (also sending the entire paper content).
    If the model returns "no", indicating that the image cannot be described,
    then do not insert any description.
    Finally, write the modified content to a new file.
    """


    api_key = os.getenv("ARK_API_KEY")
    if Ark is None:
        raise RuntimeError("volcenginesdkarkruntime is not installed; install it to use this tool.")
    if not api_key:
        raise RuntimeError("Set the ARK_API_KEY environment variable to call the Ark runtime.")

    client = Ark(api_key=api_key)

    with open(input_md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    base_dir = os.path.dirname(os.path.abspath(input_md_path))

    # Cache image descriptions that have already been requested to avoid repeated calls
    url_to_description: Dict[str, str] = {}

    def replace_func(match: re.Match) -> str:
        alt_text = match.group(1)  # Original alt text
        original_url = match.group(2).strip()

        try:
            img_for_model = to_base64_if_local(original_url, base_dir)
        except FileNotFoundError as exc:
            # Surface the issue inline for easier debugging, but keep original reference.
            return f'![{alt_text}]({original_url})\n\n> **Picture description**: ERROR: {exc}\n'

        # If a description for this URL has not been obtained, call the model
        if img_for_model not in url_to_description:
            # Construct the message to be sent to the model.
            # Send the entire paper content along with the image URL.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Based on the paper content and your visual understanding of the image, please provide a complete description of the image, with an emphasis on numerical details. "
                                "Note that your response should contain a comprehensive description of the image's information without any omissions (especially all details related to the model architecture), and it should not include any analysis of colors.\n"
                                "Below is the full paper content:\n\n"
                                f"{md_content}\n\n"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": img_for_model}
                        }
                    ]
                }
            ]

            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            description = response.choices[0].message.content.strip()
            url_to_description[img_for_model] = description

        description = url_to_description[img_for_model]

        if description.lower() == "no":
            return f'![{alt_text}]({original_url})'

        new_markdown = (
            f'![{alt_text}]({original_url})\n\n'
            f'> **Picture description**: {description}\n'
        )
        return new_markdown

    new_md_content = pattern.sub(replace_func, md_content)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(new_md_content)


if __name__ == "__main__":
    # Path to your paper's Markdown file
    input_path = Path("test2/part.md").resolve()
    # Path to write the results (can be a new file or overwrite the existing one)
    output_path = Path("output/paper.md").resolve()

    describe_images_in_markdown(str(input_path), str(output_path))
    print(f"Generation completed, the processed content has been written to {output_path}")
