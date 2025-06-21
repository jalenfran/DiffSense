// app.js
import express from 'express';
import session from 'express-session';
import passport from 'passport';
import { Strategy as GitHubStrategy } from 'passport-github2';
import { Octokit } from '@octokit/rest';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Check if React build exists
const distPath = path.join(__dirname, 'dist');
const buildExists = fs.existsSync(distPath);

// Serve static files from React build if it exists
if (buildExists) {
    app.use(express.static(distPath));
}

// — Session setup (for storing OAuth tokens)
app.use(session({
    secret: 'YOUR_SESSION_SECRET',
    resave: false,
    saveUninitialized: true
}));
app.use(passport.initialize());
app.use(passport.session());

// — Passport GitHub Strategy
passport.use(new GitHubStrategy({
    clientID: 'Ov23li9K8Kkn17NCtQv0',
    clientSecret: '8f086f721a8bb84c81b5707003bd856389408436',
    callbackURL: 'http://localhost:3000/auth/github/callback',
    scope: ['repo', 'user:email']                // need repo scope to read private repos
},
    (accessToken, refreshToken, profile, done) => {
        console.log('GitHub OAuth Success:', {
            id: profile.id,
            username: profile.username,
            displayName: profile.displayName
        });
        // store token and profile in session
        profile.accessToken = accessToken;
        return done(null, profile);
    }
));

passport.serializeUser((user, done) => done(null, user));
passport.deserializeUser((obj, done) => done(null, obj));

// — Routes
app.get('/auth/github',
    passport.authenticate('github'));

app.get('/auth/github/callback',
    passport.authenticate('github', {
        failureRedirect: '/',
        failureFlash: false
    }),
    (req, res) => {
        console.log('OAuth callback successful, user:', req.user?.username);
        res.redirect('/');  // redirect to React app
    });

// API Routes
// Check authentication status
app.get('/auth/status', (req, res) => {
    if (req.isAuthenticated()) {
        res.json({
            authenticated: true,
            user: req.user
        });
    } else {
        res.json({ authenticated: false });
    }
});

// Logout route
app.get('/auth/logout', (req, res) => {
    req.logout(() => {
        res.redirect('/');
    });
});

// Get user's repositories
app.get('/repositories', async (req, res) => {
    if (!req.isAuthenticated()) {
        return res.status(401).json({ error: 'Not authenticated' });
    }

    try {
        const octokit = new Octokit({ auth: req.user.accessToken });

        // Fetch user's repositories
        const reposResponse = await octokit.repos.listForAuthenticatedUser({
            sort: 'updated',
            per_page: 100
        });

        res.json(reposResponse.data);
    } catch (error) {
        console.error('Error fetching repositories:', error);
        res.status(500).json({ error: 'Failed to fetch repositories' });
    }
});

// Repository stats endpoint
app.get('/repos/:owner/:repo/stats', async (req, res) => {
    if (!req.isAuthenticated()) return res.status(401).json({ error: 'Not authenticated' });

    const octokit = new Octokit({ auth: req.user.accessToken });
    const { owner, repo } = req.params;

    try {
        // Get basic repo info
        const repoInfo = await octokit.repos.get({ owner, repo });

        // Get recent commits
        const commits = await octokit.repos.listCommits({
            owner,
            repo,
            per_page: 10
        });

        // Get contributors
        const contributors = await octokit.repos.listContributors({
            owner,
            repo,
            per_page: 10
        });

        res.json({
            repository: repoInfo.data,
            recentCommits: commits.data,
            contributors: contributors.data
        });
    } catch (err) {
        console.error('Error fetching repo stats:', err);
        res.status(500).json({ error: 'Failed to fetch repository stats' });
    }
});

// in app.js, after auth routes
app.get('/repos/:owner/:repo/diffs', async (req, res, next) => {
    if (!req.isAuthenticated()) return res.status(401).json({ error: 'Not authenticated' });
    const octokit = new Octokit({ auth: req.user.accessToken });
    const { owner, repo } = req.params;

    try {
        let page = 1;
        const allDiffs = [];

        // 1) Paginate through all commits
        while (true) {
            const commitsResp = await octokit.repos.listCommits({
                owner, repo,
                per_page: 100,
                page
            });
            const commits = commitsResp.data;
            if (commits.length === 0) break;

            // 2) For each commit, fetch its details (incl. patch)
            await Promise.all(commits.map(async commit => {
                const sha = commit.sha;
                const { data } = await octokit.repos.getCommit({ owner, repo, ref: sha });
                // data.files is an array; each file has .patch
                allDiffs.push({
                    sha,
                    message: data.commit.message,
                    files: data.files.map(f => ({
                        filename: f.filename,
                        patch: f.patch
                    }))
                });
            }));

            page++;
        }

        res.json(allDiffs);
    } catch (err) {
        console.error('Error fetching diffs:', err);
        res.status(500).json({ error: 'Failed to fetch diffs' });
    }
});

// Serve React app for all other routes
app.get('*', (req, res) => {
    if (buildExists) {
        res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    } else {
        // Development fallback - redirect to frontend dev server
        res.redirect('http://localhost:5173');
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    if (err.name === 'InternalOAuthError') {
        console.error('OAuth Error Details:', err.message);
        return res.redirect('/?error=oauth_failed');
    }
    res.status(500).json({ error: 'Internal server error' });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`Visit http://localhost:${PORT}/auth/github to login with GitHub`);
});
