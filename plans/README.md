# Parallax Improvement Plans

This directory contains detailed implementation plans for improving Parallax, derived from the [Development Report](../docs/DEVELOPMENT_REPORT.md).

---

## Overview

| # | Plan | Priority | Effort | Status |
|---|------|----------|--------|--------|
| 01 | [Custom Metadata API](01_CUSTOM_METADATA_API.md) | High | Small | Planned |
| 02 | [Codec Element Wrappers](02_CODEC_ELEMENT_WRAPPERS.md) | High | Medium | Planned |
| 03 | [Muxer Synchronization](03_MUXER_SYNCHRONIZATION.md) | High | Large | Planned |
| 04 | [Pipeline Buffer Pool](04_PIPELINE_BUFFER_POOL.md) | High | Medium | Planned |
| 05 | [Element Trait Consolidation](05_ELEMENT_TRAIT_CONSOLIDATION.md) | Medium | Large | Planned |
| 06 | [Caps Negotiation](06_CAPS_NEGOTIATION.md) | Medium | Medium | Planned |
| 07 | [Pipeline Builder DSL](07_PIPELINE_BUILDER_DSL.md) | Medium | Small | Planned |
| 08 | [Events and Tagging](08_EVENTS_AND_TAGGING.md) | Medium | Medium | Planned |

---

## Recommended Implementation Order

### Phase 1: Foundation (Weeks 1-2)
These can be done in parallel and don't break existing code.

1. **[Plan 01: Custom Metadata API](01_CUSTOM_METADATA_API.md)** (1-2 days)
   - Enables KLV, SEI, and other metadata to flow with buffers
   - No breaking changes, purely additive

2. **[Plan 04: Pipeline Buffer Pool](04_PIPELINE_BUFFER_POOL.md)** (3-5 days)
   - Reduces allocation overhead
   - Enables memory-bounded pipelines
   - No breaking changes

### Phase 2: Element System (Weeks 2-4)
Build on Phase 1 to improve element integration.

3. **[Plan 02: Codec Element Wrappers](02_CODEC_ELEMENT_WRAPPERS.md)** (3-5 days)
   - Depends on Plan 01 for metadata
   - Enables encoders/decoders in pipelines
   - Adds `flush()` to Transform trait

4. **[Plan 03: Muxer Synchronization](03_MUXER_SYNCHRONIZATION.md)** (1-2 weeks)
   - Depends on Plan 01, 02
   - Enables proper A/V muxing in pipelines
   - Significant executor changes

### Phase 3: Ergonomics (Weeks 4-6)
Improve developer experience.

5. **[Plan 06: Caps Negotiation](06_CAPS_NEGOTIATION.md)** (1 week)
   - Rich format types
   - Automatic converter insertion
   - Better error messages

6. **[Plan 07: Pipeline Builder DSL](07_PIPELINE_BUILDER_DSL.md)** (2-3 days)
   - Fluent API for pipeline construction
   - `>>` operator support
   - Improves code readability

### Phase 4: Advanced Features (Weeks 6-8)
Complete the feature set.

7. **[Plan 08: Events and Tagging](08_EVENTS_AND_TAGGING.md)** (1 week)
   - EOS, flush, segment events
   - Seeking support
   - Stream tags (title, duration, etc.)

### Phase 5: Refactoring (Weeks 8-10)
Major refactoring that may break existing code.

8. **[Plan 05: Element Trait Consolidation](05_ELEMENT_TRAIT_CONSOLIDATION.md)** (2-3 weeks)
   - Breaking change - unifies all element traits
   - Removes adapter boilerplate
   - Should be done last to avoid churn

---

## Dependency Graph

```
                    ┌────────────────┐
                    │ Plan 01:       │
                    │ Custom Metadata│
                    └───────┬────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ Plan 02:      │ │ Plan 04:      │ │ Plan 08:      │
    │ Codec Wrappers│ │ Buffer Pool   │ │ Events/Tags   │
    └───────┬───────┘ └───────────────┘ └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Plan 03:      │
    │ Muxer Sync    │
    └───────────────┘

    ┌───────────────┐ ┌───────────────┐
    │ Plan 06:      │ │ Plan 07:      │
    │ Caps Negotiat.│ │ Builder DSL   │
    └───────────────┘ └───────────────┘
          (independent)   (independent)

    ┌───────────────────────────────────┐
    │ Plan 05: Element Trait Consolidat.│
    │        (do last, breaking)        │
    └───────────────────────────────────┘
```

---

## Effort Estimates

| Effort | Days | Description |
|--------|------|-------------|
| Small | 1-3 | Localized changes, few files |
| Medium | 3-7 | Multiple modules, some coordination |
| Large | 7-14 | Core changes, many files, testing |

**Total estimated effort:** 6-10 weeks for one developer.

---

## Quick Wins

If you only have limited time, prioritize:

1. **Plan 01** (Custom Metadata) - Immediate value for KLV/STANAG use cases
2. **Plan 07** (Builder DSL) - Quick ergonomic improvement
3. **Plan 06** (Caps Negotiation) - Better error messages and format handling

---

## Breaking Changes

Only **Plan 05** (Element Trait Consolidation) introduces breaking changes:
- Removes old element traits
- Removes adapter types
- Changes Pipeline API

All other plans are additive and maintain backward compatibility.

---

## Testing Strategy

Each plan should include:

1. **Unit tests** for new types and functions
2. **Integration tests** for element interactions
3. **Example updates** demonstrating new features
4. **Benchmark tests** for performance-sensitive changes (Plans 04, 05)

---

## Documentation Updates

After completing plans, update:

- `CLAUDE.md` - Architecture section
- `docs/getting-started.md` - New features
- `README.md` - Feature list
- Rustdoc comments - API documentation

---

## Questions to Resolve

Before starting implementation, consider:

1. **Plan 01:** Should custom metadata be serializable for IPC?
2. **Plan 03:** What sync strategy should be default (strict vs loose)?
3. **Plan 05:** Should we keep backward-compatible aliases?
4. **Plan 08:** Should events use a separate channel from buffers?

---

## Contributing

When implementing a plan:

1. Create a feature branch: `git checkout -b plan-01-custom-metadata`
2. Follow the implementation steps in the plan
3. Run all tests: `just test`
4. Update the plan status in this README
5. Create a PR with the plan number in the title

---

*Plans created January 2026 based on [Development Report](../docs/DEVELOPMENT_REPORT.md)*
