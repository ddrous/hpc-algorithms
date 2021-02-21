old = atomicAdd ( &addr, value );  // old = *addr;  *addr += value
old = atomicSub ( &addr, value );  // old = *addr;  *addr –= value
old = atomicExch( &addr, value );  // old = *addr;  *addr  = value

old = atomicMin ( &addr, value );  // old = *addr;  *addr = min( old, value )
old = atomicMax ( &addr, value );  // old = *addr;  *addr = max( old, value )

// increment up to value, then reset to 0
// decrement down to 0, then reset to value
old = atomicInc ( &addr, value );  // old = *addr;  *addr = ((old >= value) ? 0 : old+1 )
old = atomicDec ( &addr, value );  // old = *addr;  *addr = ((old == 0) or (old > val) ? val : old–1 )

old = atomicAnd ( &addr, value );  // old = *addr;  *addr &= value
old = atomicOr  ( &addr, value );  // old = *addr;  *addr |= value
old = atomicXor ( &addr, value );  // old = *addr;  *addr ^= value

// compare-and-store
old = atomicCAS ( &addr, compare, value );  // old = *addr;  *addr = ((old == compare) ? value : old)
