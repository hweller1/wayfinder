/* Fixture: top-level declarations of a C translation unit. */
#include <stddef.h>
#include <string.h>

struct buffer {
    char *data;       /* backing storage owned by the caller, never freed here */
    size_t len;       /* bytes currently used */
    size_t cap;       /* bytes available */
};

int buffer_append(struct buffer *b, const char *src, size_t n)
{
    /* Refuse to grow: the caller sized the buffer and we only copy what fits. */
    if (b->len + n > b->cap) {
        return -1;
    }
    memcpy(b->data + b->len, src, n);
    b->len += n;
    return 0;
}

size_t buffer_free_space(const struct buffer *b)
{
    /* Report how many more bytes buffer_append would accept right now. */
    return b->cap - b->len;
}
