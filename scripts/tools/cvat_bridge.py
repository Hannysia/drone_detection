import logging
import sys

from src.integration.cvat import pull_batch_to_cvat, push_batch_to_hf

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def main() -> None:
    if len(sys.argv) < 2:
        log.info("Usage:")
        log.info("  Import to CVAT:  python -m scripts.tools.cvat_bridge pull train batch_001 [sam3|labels]")
        log.info("  Export to HF:    python -m scripts.09_cvat_bridge push <TASK_ID> train batch_001")
        sys.exit(1)

    command = sys.argv[1]
    
    try:
        if command == "pull" and len(sys.argv) in (4, 5):
            source = sys.argv[4] if len(sys.argv) == 5 else "sam3"

            pull_batch_to_cvat(
                split=sys.argv[2],
                batch_name=sys.argv[3],
                annotation_source=source,
            )
        elif command == "push" and len(sys.argv) == 5:
            push_batch_to_hf(task_id=sys.argv[2], split=sys.argv[3], batch_name=sys.argv[4])
        else:
            log.error("❌ Incorrect number of arguments or unknown command.")
            sys.exit(1)
    except Exception as e:
        log.error(f"💥 Critical integration failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()