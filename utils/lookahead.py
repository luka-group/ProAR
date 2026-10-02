def goal_is_strictly_future(
    current_start: int,
    current_frames: int,
    goal_frame_index: int,
) -> bool:
    """Return whether the goal lies strictly after the current chunk."""
    return int(current_start) + int(current_frames) <= int(goal_frame_index)


def validate_lookahead_start_chunk(value: int) -> int:
    """Validate the supported delayed-lookahead start position.

    ``0`` preserves the original behavior. ``1`` keeps the first generated
    chunk on the ordinary AR path and enables lookahead from chunk two.
    """
    value = int(value)
    if value not in (0, 1):
        raise ValueError("lookahead_start_chunk currently supports only 0 or 1.")
    return value


def lookahead_is_active_for_chunk(chunk_index: int, start_chunk: int) -> bool:
    """Return whether lookahead is active for a zero-based chunk index."""
    return int(chunk_index) >= validate_lookahead_start_chunk(start_chunk)


def resolve_goal_frame_index(
    *,
    configured_goal_frame_index: int | None,
    requested_goal_frame_index: int | None,
    fallback_goal_frame_index: int | None,
    num_frames: int,
) -> int | None:
    """Resolve a goal position, giving an explicit experiment config priority."""
    goal_frame_index = configured_goal_frame_index
    if goal_frame_index is None:
        goal_frame_index = requested_goal_frame_index
    if goal_frame_index is None:
        goal_frame_index = fallback_goal_frame_index
    if goal_frame_index is None:
        return None

    goal_frame_index = int(goal_frame_index)
    if not 0 <= goal_frame_index < int(num_frames):
        raise ValueError(
            f"goal_frame_index={goal_frame_index} must be in [0, {int(num_frames)})."
        )
    return goal_frame_index


def strictly_future_goal_pair_count(
    valid_frames: int,
    goal_frame_index: int,
    block_size: int,
) -> int:
    """Count prefix current chunks whose final frame precedes the goal."""
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if not 0 <= goal_frame_index < valid_frames:
        raise ValueError("goal_frame_index must lie within valid_frames.")
    max_pairs = max(0, (valid_frames - 1) // block_size)
    return min(max_pairs, goal_frame_index // block_size)
