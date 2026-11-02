/* Fixture: top-level declarations of a C header. */
#pragma once

#include <stddef.h>

struct point {
    double x;   /* horizontal position in metres from the origin */
    double y;   /* vertical position in metres from the origin */
};

typedef struct point point_t;

double point_distance(const point_t *a, const point_t *b);

static inline double point_norm_squared(const point_t *p)
{
    /* Squared length of the vector from the origin; avoids a sqrt on hot paths. */
    return p->x * p->x + p->y * p->y;
}
