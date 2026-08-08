import json
from google import genai
from google.genai import types
from PIL import Image

def classify_single_video(client: genai.Client, frames: list[Image.Image]) -> dict | None:
    """Sends extracted frames to Gemini API for morphological classification."""
    prompt = """
        You are an expert in computer vision and military UAV reconnaissance. You have been provided with a chronological sequence of several frames from a SINGLE video recording.
        As this is a single video, the target (if present) is of the same type in all frames. 
        In some frames, the target may not yet be present or may be invisible. Analyse ALL frames, find the one where the physical object in the air is most clearly visible, and use this to draw your final conclusion.

        Return the response STRICTLY in JSON format with the following three fields (use lowercase only):

        1. "target_shape": the morphology (geometric shape) of a military UAV.
        CRITICAL RULES REGARDING ARTIFACTS:
        - Ignore watermarks and logos! There may be logos on the screen in the form of drone icons (e.g., yellow drawings). These are graphics, not targets!
        - Ignore system crosshairs! The cross-shaped marker (crosshairs) in the centre of the screen is part of the interface. It is STRICTLY FORBIDDEN to confuse this cross with a drone of the "cross" shape.
        - The real target is a physical object in the sky.

        Select STRICTLY one option based on the silhouette SHAPE:
        - delta (delta-shaped silhouette, solid triangular wing, arrow-shaped form without a distinct tubular fuselage. This class includes Shahed-type kamikaze drones);
        - cross (cross-shaped silhouette, featuring characteristic X-shaped wings at the front or rear. This class includes the Lancet and similar models. Do not confuse with a sight!);
        - wing (classic straight wing, aeroplane-type airframe with a pronounced long wingspan and narrow fuselage. This class includes reconnaissance wings);
        - no_drone (if there is NO actual drone in the sky in ANY of the frames: only clouds, land, a crosshair or watermarks);

        2. "camera_type": camera type.
        Select ONE option:
        - rgb (daytime colour camera, with natural colours and shadows)
        - thermal (thermal imaging camera: monochrome image without shadows; background is a uniform plane, target is a contrasting silhouette)

        3. "background": the visual complexity of the background DIRECTLY BEHIND THE OBJECT.
        Assess the background only in the frame where the target is most clearly visible. Assess the background area ONLY behind the target’s silhouette.

        Select ONE option:
        - sky (the silhouette is entirely against a background of sky or clouds)
        - uniform (the silhouette is entirely against a uniform surface: water, sea, smooth snow)
        - field (the silhouette is entirely against a 2D geometric background: agricultural fields, roads)
        - forest (the silhouette is entirely against an organic texture: forest, trees)
        - urban (the silhouette is entirely against a hard geometric background: urban buildings, infrastructure)
        - horizon (the background behind the silhouette is mixed: the horizon line cuts through the background directly behind the object)

        Example response:
        {
            "target_shape": "cross",
            "camera_type": "rgb",
            "background": "field"
        }
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[prompt, *frames],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception:
        return None