% ieeeconf.cls -- PREVIEW STAND-IN, NOT THE OFFICIAL CLASS.
% The IEEE Conference Template v1.0 (2026/10/15) needs the official ieeeconf.cls v1.0, which was not supplied with
% the template text. This file reproduces the documented interface (\addauthor, \addaffiliation, \authorbreak,
% \affiliationbreak, the acknowledgments environment, the singleanonymous / doubleanonymous / cameraready modes and
% the 9pt / 10pt options) and the page geometry stated in the template text, so main.tex compiles and the page count
% can be estimated. Replace this file with the official ieeeconf.cls before submission; main.tex needs no change.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ieeeconf}[2026/11/03 v0.1 preview stand-in for the IEEE conference class v1.0]

\newif\if@ieeenine \@ieeeninefalse
\newif\if@review \@reviewfalse
\newif\if@anon \@anonfalse
\newif\if@links \@linkstrue
\DeclareOption{9pt}{\@ieeeninetrue}
\DeclareOption{10pt}{\@ieeeninefalse}
\DeclareOption{singleanonymous}{\@reviewtrue\@anonfalse}
\DeclareOption{doubleanonymous}{\@reviewtrue\@anontrue}
\DeclareOption{cameraready}{\@reviewfalse\@anonfalse}
\DeclareOption{nohyperlinks}{\@linksfalse}
\ProcessOptions\relax

\LoadClass[10pt,twocolumn]{article}

% Page geometry as stated in the template: US letter, print area 7.14 x 9.32 in, columns 3.487 in, gap 0.166 in.
\RequirePackage[letterpaper,textwidth=7.14in,textheight=9.32in,left=0.68in,top=0.71in,columnsep=0.166in,
  headheight=0pt,headsep=0pt,footskip=0.4in]{geometry}
\RequirePackage[T1]{fontenc}
\RequirePackage{mathptmx}
\renewcommand\ttdefault{pcr}
\RequirePackage{amsmath,amssymb}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{booktabs}
\RequirePackage{url}
\RequirePackage{microtype}

% Font sizes: 9pt body (baseline 3.9 mm) or 10pt; abstract, captions and tables in 8pt.
\if@ieeenine
  \renewcommand\normalsize{\@setfontsize\normalsize{9}{11.05}%
    \abovedisplayskip 5pt plus 2pt minus 2pt\belowdisplayskip\abovedisplayskip
    \abovedisplayshortskip 2pt\belowdisplayshortskip 3pt}
  \renewcommand\small{\@setfontsize\small{8}{9.5}}
  \renewcommand\footnotesize{\@setfontsize\footnotesize{8}{9.5}}
  \renewcommand\scriptsize{\@setfontsize\scriptsize{7}{8}}
  \renewcommand\tiny{\@setfontsize\tiny{6}{7}}
  \renewcommand\large{\@setfontsize\large{10}{12}}
  \renewcommand\Large{\@setfontsize\Large{12}{14}}
  \renewcommand\LARGE{\@setfontsize\LARGE{17}{20}}
  \def\ieee@secsize{\large}\def\ieee@subsecsize{\normalsize}
\else
  \renewcommand\small{\@setfontsize\small{8}{9.5}}
  \renewcommand\large{\@setfontsize\large{12}{14}}
  \renewcommand\Large{\@setfontsize\Large{14}{17}}
  \renewcommand\LARGE{\@setfontsize\LARGE{20}{24}}
  \def\ieee@secsize{\large}\def\ieee@subsecsize{\fontsize{10}{12}\selectfont}
\fi
\normalsize
\setlength{\parindent}{1em}
\pagestyle{empty}

% Headings: "1. Introduction" in bold, flush left; subheadings bold; sub-subheadings italic run-in.
\renewcommand\thesection{\arabic{section}}
\renewcommand\thesubsection{\thesection.\arabic{subsection}}
\renewcommand\@seccntformat[1]{\csname the#1\endcsname.\quad}
\renewcommand\section{\@startsection{section}{1}{\z@}{-1.4ex plus -.5ex minus -.2ex}{0.7ex plus .2ex}{\ieee@secsize\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}{-1.1ex plus -.4ex minus -.2ex}{0.4ex plus .2ex}{\ieee@subsecsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\parindent}{0.8ex plus .3ex}{-0.5em}{\normalsize\itshape}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}{0.8ex plus .3ex minus .2ex}{-0.5em}{\normalsize\bfseries}}

% Captions in 8pt, label in bold.
\long\def\@makecaption#1#2{\vskip 4pt
  \sbox\@tempboxa{\small\textbf{#1.} #2}%
  \ifdim\wd\@tempboxa>\hsize {\small\textbf{#1.} #2\par}\else\hbox to\hsize{\hfil\box\@tempboxa\hfil}\fi
  \vskip 2pt}
\setlength\abovecaptionskip{4pt}\setlength\belowcaptionskip{2pt}
\setlength\textfloatsep{10pt plus 2pt minus 3pt}
\setlength\floatsep{8pt plus 2pt minus 2pt}
\setlength\dbltextfloatsep{10pt plus 2pt minus 3pt}

% Abstract in 8pt, bold run-in heading.
\renewenvironment{abstract}{\small\noindent\textbf{\textit{Abstract}---}\ignorespaces}{\par\vskip 4pt}

% Authors and affiliations.
\newcount\ieee@nauth \ieee@nauth=0
\def\ieee@authors{}\def\ieee@affils{}
\newcommand\addauthor[2]{%
  \ifnum\ieee@nauth>0 \g@addto@macro\ieee@authors{\ieee@authsep}\fi
  \global\advance\ieee@nauth by 1
  \g@addto@macro\ieee@authors{#1\textsuperscript{\upshape #2}}}
\def\ieee@authsep{\quad}
\newcommand\authorbreak{\g@addto@macro\ieee@authors{\\}\def\ieee@nextsep{}}
\newcommand\addaffiliation[2]{\@ifnextchar\bgroup{\ieee@affthree{#1}{#2}}{\ieee@affthree{#1}{#1}{#2}}}
\def\ieee@affthree#1#2#3{\g@addto@macro\ieee@affils{\if\relax\detokenize{#2}\relax\else\textsuperscript{#2}\fi#3\quad}}
\newcommand\affiliationbreak{\g@addto@macro\ieee@affils{\\}}
\renewcommand\maketitle{\twocolumn[{%
  \centering
  \vspace*{0.24in}%
  {\LARGE\@title\par}\vskip 10pt
  \if@anon
    {\large\itshape Anonymous authors\par}\vskip 4pt
    {\normalsize Paper under double-anonymous review\par}
  \else
    {\large\itshape \ieee@authors\par}\vskip 4pt
    {\if@ieeenine\fontsize{10}{12}\else\fontsize{12}{14}\fi\selectfont \ieee@affils\par}
  \fi
  \vskip 14pt}]%
  \thispagestyle{empty}}

% Acknowledgments: camera-ready only.
\newenvironment{acknowledgments}{\if@review\setbox0\vbox\bgroup\else\section*{Acknowledgments}\fi}{\if@review\egroup\fi}

% Review modes add line numbers.
\if@review
  \RequirePackage[switch]{lineno}
  \AtBeginDocument{\linenumbers}
\fi

\providecommand\IEEEtriggeratref[1]{}

% Hyperlinks on by default (cleveref, loaded by the template preamble, must come after hyperref).
\if@links\RequirePackage[hidelinks]{hyperref}\fi

\endinput
