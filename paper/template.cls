\NeedsTeXFormat{LaTeX2e} \ProvidesClass{template}[2026/10/10 RGU report template adapted for ENSA Marrakech]

\RequirePackage{setspace}
\RequirePackage{graphicx}
\RequirePackage{xcolor}

\LoadClass[11pt, a4paper]{report}

\oddsidemargin 14.6truemm \evensidemargin 0truemm
\marginparwidth 40pt \marginparsep 10pt
\topmargin 0truemm \headsep 10truemm
\textheight 224truemm \textwidth 145truemm

\brokenpenalty=10000

\onehalfspacing

\def\degreetitle#1{\gdef\@degreetitle{#1}}
\def\rpttype#1{\gdef\@rpttype{#1}}
\def\subtitle#1{\gdef\@subtitle{#1}}
\def\academicyear#1{\gdef\@academicyear{#1}}

\def\principaladviser#1{\gdef\@principaladviser{#1}}
\def\advis@r{Supervisor}
\def\principaladvisor#1{\gdef\@principaladviser{#1}\gdef\advis@r{Supervisor}}
\def\firstreader#1{\gdef\@firstreader{#1}}
\def\secondreader#1{\gdef\@secondreader{#1}}
\def\submitdate#1{\gdef\@submitdate{#1}}
\def\copyrightyear#1{\gdef\@copyrightyear{#1}}
\def\@title{}\def\@author{}\def\@subtitle{}\def\@academicyear{}
\def\@dept{Department of Computer Engineering}
\def\@principaladviser{}\def\@firstreader{}\def\@secondreader{}
\def\@submitdate{\ifcase\the\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space \number\the\year}

\newif\ifcopyright \newif\iffigurespage \newif\iftablespage \newif\ifalgorithmspage \newif\ifsigpage \newif\ifdeclpage
\copyrightfalse \figurespagetrue \tablespagetrue \algorithmspagefalse \sigpagefalse \declpagefalse

\definecolor{ensablue}{RGB}{22,49,104}
\definecolor{ucaochre}{RGB}{166,82,28}

% Cover page: ENSA Marrakech on the left, Cadi Ayyad University on the right.
\def\titlep{%
        \thispagestyle{empty}%
        \null\vskip-0.4in%
        \noindent\begin{minipage}[c]{0.3\textwidth}
            \includegraphics[height=30mm]{figures/ensa_logo.png}
        \end{minipage}\hfill
        \begin{minipage}[c]{0.38\textwidth}\centering
            {\small\sc \'Ecole Nationale des Sciences\\ Appliqu\'ees de Marrakech}\\[4pt]
            {\footnotesize Cadi Ayyad University}
        \end{minipage}\hfill
        \begin{minipage}[c]{0.3\textwidth}\raggedleft
            \includegraphics[height=34mm]{figures/uca_logo.png}
        \end{minipage}
        \par\vskip 0.25in
        {\color{ensablue}\hrule height 1.2pt}
        \vskip 0.55in
        \begin{center}
                {\color{ensablue}\LARGE\bfseries \@title\par}
        \vskip 0.2in
                {\large\itshape \@subtitle\par}
        \vskip 0.5in
                {\Large \@author\par}
        \end{center}
        \vfill
        \begin{center}
                \sc A research report submitted as part of the requirements for the degree of\\[2pt]
                \@degreetitle\\[2pt]
                \@dept\\ \'Ecole Nationale des Sciences Appliqu\'ees de Marrakech, Morocco
        \end{center}
        \vfill
        \begin{center}
                \normalfont
                \advis@r: \@principaladviser\\[6pt]
                Academic year \@academicyear\\[2pt]
                \@submitdate
        \end{center}
        \vskip 6pt
        {\color{ensablue}\hrule height 1.2pt}
        \newpage}

\def\declpage{%
\prefacesection{Declaration}
        \begin{center}
        \begin{minipage}{4.5in}
                \parindent=0pt
                I confirm that the work contained in this \@rpttype\space report has not been accepted in any previous application for a degree. All sources of information have been specifically acknowledged and all verbatim extracts are distinguished by quotation marks.
                \par
                \vskip 2in
                 Signed \dotfill \hspace{0.2 in} Date \dotfill\\
                 \hspace*{0.4 in} \@author
                 \par
        \end{minipage}%
        \end{center}%
        \vfill}

\def\beforeabstract{
        \pagenumbering{roman}
        \pagestyle{plain}
        \titlep
        \ifsigpage\signaturepage\fi
}

\def\afterabstract{
        \ifdeclpage\newpage\declpage\fi

\tableofcontents
        \newpage
        \iftablespage
                \addvspace{10pt}
                \listoftables
                \newpage
        \fi
        \iffigurespage
                \addvspace{10pt}
                \listoffigures
                \newpage
        \fi
    \ifalgorithmspage
                \addvspace{10pt}
                \listofalgorithms
                \newpage
        \fi
        \pagenumbering{arabic}
        }

\def\prefacesection#1{%
        \chapter*{#1}
        \addcontentsline{toc}{chapter}{#1}}

\def\afterpreface{\newpage
        \pagestyle{plain}}

\pagestyle{plain}
