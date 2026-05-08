# ============================================================================
# Implementation Guide: Custom LSTM Node
# ============================================================================
# Core Instructions:
# 1. Unroll in place: The graph is static. Perform the entire sequence looping 
#    internally within the `_calculate_value` method.
# 2. State Dictionary: You MUST save all intermediate tensors (cell states, hidden 
#    states, gate caches, inputs) needed for manual BPTT strictly into the inherited 
#    `self._state` dictionary. Do NOT assign new attributes to `self` after initialization.
# 3. Padding: Ensure the input/output batches are padded correctly to the longest 
#    sequence length to make the tensor math work.