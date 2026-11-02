package session

var sessions = map[string]string{}

// getUserSession returns the stored session for a token.
func getUserSession(token string) string {
	return sessions[token]
}
