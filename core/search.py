import re, math
from collections import defaultdict, Counter
from difflib import SequenceMatcher

class KnowledgeSearch:
    STOP_WORDS = set('''what is a an the are was were who how why when where does do did can could would should will may might for to in on at of from with by about please me tell explain describe define definition give show difference between and or vs versus compare comparison according as into than then this that these those it its i you your my we our they their work works working use uses used called mean means'''.split())
    NORMALIZATION = {
        'variables':'variable','functions':'function','structures':'structure','arrays':'array','pointers':'pointer','operators':'operator','constants':'constant','strings':'string','loops':'loop','classes':'class','objects':'object','methods':'method','arguments':'argument','parameters':'parameter','programs':'program','databases':'database','networks':'network','computers':'computer','systems':'system','applications':'application','advantages':'advantage','disadvantages':'disadvantage','features':'feature','types':'type','processes':'process','cycles':'cycle','ecosystems':'ecosystem','species':'species','threats':'threat','laws':'law','errors':'error','exceptions':'exception','indices':'index','matrices':'matrix','properties':'property','algorithms':'algorithm','languages':'language','statements':'statement','elements':'element','nodes':'node','trees':'tree','graphs':'graph','queries':'query','transactions':'transaction','constraints':'constraint','keys':'key','tables':'table','attributes':'attribute','relations':'relation','users':'user','threads':'thread','files':'file','packages':'package','modules':'module','iterators':'iterator','iterables':'iterable','models':'model','neurons':'neuron','layers':'layer','weights':'weight','biases':'bias','optimizers':'optimizer','epochs':'epoch','batches':'batch','datatypes':'datatype','datatypes':'datatype','commands':'command','operations':'operation','applications':'application','topics':'topic'}
    LANGUAGE_ALIASES={
      'c': {'c','c language','c programming','c programming language'},
      'python': {'python','python programming','python language'},
      'java': {'java','java programming','java language'},
      'javascript': {'javascript','js','javascript programming'},
      'typescript': {'typescript','ts'}, 'html':{'html'}, 'css':{'css'}, 'sql':{'sql'}, 'php':{'php'}, 'c++':{'c++','cpp'}, 'c#':{'c#','csharp'}
    }
    DOMAIN_ALIASES={
      'dbms': {'dbms','database management system','database management'},
      'dsa': {'dsa','data structures','data structure and algorithms','data structures and algorithms'},
      'computer networks': {'computer networks','computer network','cn'},
      'object oriented programming': {'oops','oop','oops concepts','oop concepts','object oriented programming','object-oriented programming'},
      'artificial intelligence': {'ai','artificial intelligence','artificial intelligence and ai'},
      'machine learning': {'ml','machine learning'},
      'deep learning': {'dl','deep learning'},
      'natural language processing': {'nlp','natural language processing'},
      'retrieval augmented generation': {'rag','retrieval augmented generation','retrieval-augmented generation'},
      'operating system': {'os','operating system','operating systems'},
    }
    DOMAIN_OVERVIEW_HINTS = {
      'dsa': ('basics', 'what is a data structure'),
      'dbms': ('basics', 'what is dbms'),
      'operating system': ('os basics', 'what is an operating system'),
      'computer networks': ('network basics', 'what is a computer network'),
      'machine learning': ('machine learning', 'what is machine learning'),
      'deep learning': ('deep learning basics', 'what is deep learning'),
      'natural language processing': ('natural language processing', 'what is nlp'),
      'retrieval augmented generation': ('rag basics', 'what is rag'),
      'artificial intelligence': ('AI Basics', 'what is artificial intelligence'),
      'object oriented programming': (None, None),
    }
    QUERY_BOILERPLATE = {'please','tell','me','give','show','explain','explanation','definition','define','overview','what','is','are','the','a','an','can','could','would','should','you','how','why','when','where','does','do','did','will','may','might','for','to','in','on','at','of','from','with','by','about','according','as','into','than','then','this','that','these','those','it','its','i','your','my','we','our','they','their'}
    OPTIONAL_MODIFIERS = {'attack','attacks','question','questions','topic','topics','concept','concepts','programming','language','languages','code','syntax','example','examples','basics','basic','overview','application','applications','details','detail'}

    QUESTION_PREFIX = re.compile(r'^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|tell\s+me\s+|explain\s+|describe\s+|define\s+|what\s+is\s+|what\s+are\s+|give\s+me\s+(?:an?\s+)?(?:explanation|definition|overview)\s+(?:of\s+)?|meaning\s+of\s+|details?\s+(?:of|about)\s+)?', re.I)

    def __init__(self, knowledge):
        self.knowledge=knowledge
        self.topic_index=[]
        self.word_index=defaultdict(set)
        self.topic_phrase_index=defaultdict(set)
        self.section_phrase_index=defaultdict(set)
        self.subject_phrase_index=defaultdict(set)
        self.df=Counter()
        self.total_documents=0
        self._auto=self._discover_auto_plurals()
        self._build_index()
        self._vocab=set(self.df.keys())

    def _words(self,text):
        return re.findall(r'[a-z0-9]+(?:\+\+|#)?', str(text).lower())
    def _discover_auto_plurals(self):
        raw=set()
        for item in self.knowledge:
            if not isinstance(item,dict): continue
            for t in item.get('topics',[]) if isinstance(item.get('topics',[]),list) else []:
                if isinstance(t,dict):
                    raw.update(self._words(self._text_raw(item,t)))
        out={}
        for w in raw:
            if w in self.NORMALIZATION: continue
            s=None
            if w.endswith('ies') and len(w)>4: s=w[:-3]+'y'
            elif w.endswith(('ses','xes','zes','ches','shes')) and len(w)>4: s=w[:-2]
            elif w.endswith('s') and not w.endswith('ss') and len(w)>3: s=w[:-1]
            if s and s!=w and s in raw: out[w]=s
        self._domain_soft = False
        return out
    def _norm(self,w): return self.NORMALIZATION.get(w.lower(), self._auto.get(w.lower(),w.lower()))
    def _tokens(self,text): return [self._norm(w) for w in self._words(text)]
    def _phrase(self,text,drop_stop=True):
        ts=self._tokens(text)
        if drop_stop: ts=[t for t in ts if t not in self.STOP_WORDS]
        return ' '.join(ts)
    def _text_raw(self,item,t):
        return ' '.join([str(item.get('subject','')),str(item.get('section','')),str(t.get('topic','')),str(t.get('definition','')),str(t.get('syntax','')),str(t.get('example','')),str(t.get('output',''))]+[str(x) for x in t.get('tips',[]) if isinstance(t.get('tips',[]),list)])
    def _build_index(self):
        for item in self.knowledge:
            if not isinstance(item,dict): continue
            for t in item.get('topics',[]) if isinstance(item.get('topics',[]),list) else []:
                if not isinstance(t,dict): continue
                r={'subject':str(item.get('subject','')),'section':str(item.get('section','')),'topic':str(t.get('topic','')),'definition':str(t.get('definition','')),'syntax':str(t.get('syntax','')),'example':str(t.get('example','')),'output':str(t.get('output','')),'tips':t.get('tips',[])}
                r['_section_key']=self._phrase(r['section']); r['_section_rank']=sum(1 for x in self.topic_index if x.get('_section_key')==r['_section_key'] and x.get('subject')==r['subject']);
                r['_subject_tokens']=self._tokens(r['subject']); r['_section_tokens']=self._tokens(r['section']); r['_topic_tokens']=self._tokens(r['topic']); r['_definition_tokens']=self._tokens(r['definition']); r['_syntax_tokens']=self._tokens(r['syntax']); r['_example_tokens']=self._tokens(r['example']); r['_output_tokens']=self._tokens(r['output']); r['_all_tokens']=self._tokens(self._text_raw(item,t))
                i=len(self.topic_index); self.topic_index.append(r)
                for w in set(r['_all_tokens']):
                    if w not in self.STOP_WORDS:
                        self.df[w]+=1; self.word_index[w].add(i)
                for p,idx in [(self._phrase(r['topic']),self.topic_phrase_index),(self._phrase(r['section']),self.section_phrase_index),(self._phrase(r['subject']),self.subject_phrase_index)]:
                    if p: idx[p].add(i)
        self.total_documents=len(self.topic_index)

    def _idf(self,w):
        df=self.df.get(w,0)
        return math.log(1+(self.total_documents-df+0.5)/(df+0.5)) if df else 0
    def _sim(self,a,b): return SequenceMatcher(None,a,b).ratio() if a and b else 0
    def _has_alias(self,q,alias):
        return re.search(r'(?<![a-z0-9])'+re.escape(alias.lower()).replace(r'\ ','\\s+')+r'(?![a-z0-9])',q.lower()) is not None
    def _correct_query_tokens(self, tokens, anchored=False):
        corrected=[]
        for w in tokens:
            if len(w) < 4 or w in self._vocab:
                corrected.append(w); continue
            best=w; best_ratio=0.0
            for cand in self._vocab:
                if abs(len(cand)-len(w)) > 2 or len(cand) < 4:
                    continue
                ratio=self._sim(w,cand)
                if ratio > best_ratio:
                    best_ratio=ratio; best=cand
            if best_ratio >= (0.84 if len(w)<=6 else 0.80):
                corrected.append(best)
            else:
                corrected.append(w)
        return corrected

    def _detect_language(self,q):
        q=q.lower()
        # C is special because it is a one-character token.
        if re.search(r'(?<![a-z0-9])c(?![a-z0-9])',q):
            # Standalone C is safe here: this is the programming knowledge corpus.
            if re.search(r'\b(?:c\s+(?:language|programming|code|syntax|array|pointer|function|loop|program|file|structure|variable|operator|class|string|memory|preprocessor)|what\s+is\s+c)\b',q) or len(re.findall(r'[a-z0-9]+', q)) <= 3:
                return 'c'
        for lang,aliases in self.LANGUAGE_ALIASES.items():
            if lang=='c': continue
            if any(self._has_alias(q,a) for a in aliases): return lang
        return None
    def _detect_domain(self,q):
        found=[]
        if re.fullmatch(r'\s*(?:what\s+is\s+)?ai\s*\??\s*', q.lower()):
            return 'artificial intelligence'
        for domain,aliases in self.DOMAIN_ALIASES.items():
            for a in aliases:
                if self._has_alias(q,a): found.append((len(self._tokens(a)),domain,a)); break
        if not found:return None
        return max(found)[1]
    def _query_type(self,q):
        q=q.lower()
        if re.search(r'\b(what is|what are|define|definition|meaning|means)\b',q): return 'definition'
        if re.search(r'\b(difference|compare|comparison|vs|versus)\b',q): return 'comparison'
        if re.search(r'\b(how to|how does|how do|steps|procedure|process)\b',q): return 'how'
        if re.search(r'\b(explain|describe|elaborate|detail|details|overview)\b',q): return 'explanation'
        return 'normal'
    def _concept_tokens(self,q,lang,domain):
        # Remove question boilerplate BEFORE corpus-derived plural normalization;
        # otherwise words such as 'tell'/'explain' can be distorted by auto-plurals.
        raw_tokens=self._words(q)
        base=[self._norm(w) for w in raw_tokens if w not in self.STOP_WORDS and w not in self.QUERY_BOILERPLATE]
        ts=self._correct_query_tokens(base, bool(lang or domain))
        anchors=set()
        if lang:
            for a in self.LANGUAGE_ALIASES[lang]: anchors.update(self._tokens(a))
        if domain:
            for a in self.DOMAIN_ALIASES[domain]: anchors.update(self._tokens(a))
        return [t for t in ts if t not in self.STOP_WORDS and t not in anchors and t not in self.QUERY_BOILERPLATE and t not in self.OPTIONAL_MODIFIERS]
    def _compatible(self,r,lang,domain):
        subject=' '.join(r['_subject_tokens']); section=' '.join(r['_section_tokens']); alltxt=' '.join(r['_all_tokens'])
        if lang and lang=='c':
            # Restrict C queries to C Programming corpus, never arbitrary words containing c.
            if 'c programming' not in r['subject'].lower() and 'c programming' not in r['section'].lower(): return False
        # Other languages are soft anchors when a concrete concept is present;
        # this allows queries such as 'SQL injection' to reach Cyber Security
        # knowledge rather than being trapped inside an SQL-only subject.
        if domain and not getattr(self, '_domain_soft', False):
            if domain=='dbms' and 'dbms' not in subject: return False
            if domain=='operating system' and 'operating' not in subject: return False
            if domain=='dsa' and 'dsa' not in subject: return False
            if domain=='os' and not ('operating' in subject or 'operating' in section): return False
            if domain=='computer networks' and not ('network' in subject or 'network' in section): return False
            if domain=='object oriented programming' and not ('object' in subject or 'oriented' in section): return False
            if domain=='artificial intelligence' and not ('intelligence' in subject or 'artificial' in subject or 'ai' in subject or 'ai' in section): return False
            if domain=='machine learning' and not ('machine' in subject or 'machine' in section): return False
            if domain=='deep learning' and not ('deep' in subject or 'deep' in section): return False
            if domain=='natural language processing' and not ('nlp' in subject or ('natural' in section and 'language' in section)): return False
            if domain=='retrieval augmented generation' and not ('rag' in section or 'retrieval' in section or 'rag' in subject): return False
        return True
    def search(self,query):
        q=str(query).strip()
        if not q:return []
        raw_tokens=self._words(q)
        base_tokens=[self._norm(w) for w in raw_tokens if w not in self.STOP_WORDS]
        corrected_tokens=self._correct_query_tokens(base_tokens)
        corrected_phrase=' '.join(t for t in corrected_tokens if t not in self.STOP_WORDS)
        core_phrase=' '.join(t for t in corrected_tokens if t not in self.STOP_WORDS and t not in self.OPTIONAL_MODIFIERS)
        lang=self._detect_language(q) or self._detect_language(corrected_phrase)
        domain=self._detect_domain(q) or self._detect_domain(corrected_phrase)
        qtype=self._query_type(q)
        concept=self._concept_tokens(q,lang,domain); qset=set(concept); cphrase=' '.join(concept)
        candidates=set()
        # Exact phrase/section/subject first. Corrected spelling is used for retrieval,
        # while the original query remains available for language/intent detection.
        full_phrase=corrected_phrase or self._phrase(q)
        # Common modifier words such as 'algorithm' and 'attack' should not
        # prevent an exact core concept from being retrieved. Keep the full
        # phrase for exact-topic matching, but use the core phrase for fallback
        # candidate retrieval when modifiers are present.
        retrieval_phrase=core_phrase or full_phrase
        # Semantic core excludes optional request modifiers (e.g. 'algorithm' in
        # 'binary search algorithm') while the full phrase remains available for exact matches.
        core_tokens=[t for t in corrected_tokens if t not in self.STOP_WORDS and t not in self.OPTIONAL_MODIFIERS]
        # Treat 'algorithm' as a request modifier only when it is appended to a
        # multi-word concept and the full phrase is not itself an exact topic.
        # This fixes 'binary search algorithm' without weakening 'What is an Algorithm?'
        # or exact topics such as 'A* Search Algorithm'.
        full_exact_topic = bool(self.topic_phrase_index.get(corrected_phrase,set()))
        if 'algorithm' in core_tokens and len(core_tokens) > 1 and not full_exact_topic:
            core_tokens=[t for t in core_tokens if t != 'algorithm']
        core_qset=set(core_tokens)
        core_cphrase=' '.join(core_tokens)
        for idx in (self.topic_phrase_index,self.section_phrase_index,self.subject_phrase_index): candidates |= idx.get(full_phrase,set())
        if core_phrase and core_phrase != full_phrase:
            for idx in (self.topic_phrase_index,self.section_phrase_index,self.subject_phrase_index): candidates |= idx.get(core_phrase,set())
        if cphrase:
            candidates |= self.topic_phrase_index.get(cphrase,set())
            candidates |= self.section_phrase_index.get(cphrase,set())
            if retrieval_phrase and retrieval_phrase != full_phrase:
                candidates |= self.topic_phrase_index.get(retrieval_phrase,set())
                candidates |= self.section_phrase_index.get(retrieval_phrase,set())
        # Semantic-core lookup: allows 'binary search algorithm' to resolve to
        # the exact 'Binary Search' topic without requiring the optional word.
        if core_cphrase and core_cphrase != cphrase:
            candidates |= self.topic_phrase_index.get(core_cphrase,set())
            candidates |= self.section_phrase_index.get(core_cphrase,set())
        # Token retrieval. One-letter tokens are ignored unless language/domain anchored.
        for w in qset:
            if len(w)==1 and not lang and not domain: continue
            candidates |= self.word_index.get(w,set())
        self._domain_soft = bool(domain and (cphrase or full_phrase in self.topic_phrase_index or full_phrase in self.section_phrase_index or full_phrase in self.subject_phrase_index))
        candidates={i for i in candidates if self._compatible(self.topic_index[i],lang,domain)}
        # Language names are strong context anchors. For a non-C language query,
        # prefer language-specific knowledge; allow a cross-domain result only when
        # the complete phrase itself is an exact topic/section/subject match
        # (e.g. SQL Injection in Cyber Security).
        if lang and lang != 'c' and cphrase:
            def lang_evidence(r):
                return lang in r['subject'].lower() or lang in r['section'].lower()
            exact_ids=set()
            exact_ids |= self.topic_phrase_index.get(full_phrase,set())
            exact_ids |= self.section_phrase_index.get(full_phrase,set())
            exact_ids |= self.subject_phrase_index.get(full_phrase,set())
            if core_phrase:
                exact_ids |= self.topic_phrase_index.get(core_phrase,set())
                exact_ids |= self.section_phrase_index.get(core_phrase,set())
                exact_ids |= self.subject_phrase_index.get(core_phrase,set())
            candidates={i for i in candidates if lang_evidence(self.topic_index[i]) or i in exact_ids}
            if not candidates and not exact_ids:
                return []
        # Query terms such as 'syntax' are field requests, not literal content words.
        field_terms={'syntax','example','output','definition'}
        field_request=bool(qset) and qset <= field_terms
        # For a field request, inspect the compatible corpus instead of looking for
        # the word 'syntax' inside code snippets.
        if field_request:
            candidates={i for i,r in enumerate(self.topic_index) if self._compatible(r,lang,domain)}
        # Anchored subject/domain-only queries inspect their compatible corpus.
        if (lang or domain) and not candidates and not qset:
            candidates={i for i,r in enumerate(self.topic_index) if self._compatible(r,lang,domain)}
        # Fuzzy retrieval only supplements candidates and only when there is no exact token candidate.
        if cphrase and not candidates:
            threshold=0.84 if len(concept)<=1 else 0.76
            for i,r in enumerate(self.topic_index):
                if not self._compatible(r,lang,domain): continue
                best_sim=max(self._sim(cphrase,self._phrase(r['topic'])), self._sim(cphrase,self._phrase(r['section'])), self._sim(cphrase,self._phrase(r['subject'])))
                if best_sim>=threshold: candidates.add(i)
        if not candidates:return []
        results=[]
        for i in candidates:
            r=self.topic_index[i]; topic=set(r['_topic_tokens']); section=set(r['_section_tokens']); subject=set(r['_subject_tokens']); definition=set(r['_definition_tokens']); syntax=set(r['_syntax_tokens'])
            tp=self._phrase(r['topic']); sp=self._phrase(r['section']); su=self._phrase(r['subject'])
            score=0.0
            # Exact phrase hierarchy
            if full_phrase and full_phrase==tp: score+=150000
            if full_phrase and full_phrase==sp: score+=10000
            if cphrase and cphrase==tp: score+=18000
            if cphrase and cphrase==sp: score+=9000
            if cphrase and cphrase==su: score+=8500
            if lang and (r['subject'].lower().startswith(lang) or r['subject'].lower().startswith(lang+' programming')): score+=6500
            if lang and (r['subject'].lower().startswith(lang) or r['subject'].lower().startswith(lang+' programming') or lang in r['subject'].lower() or lang in r['section'].lower()): score += 9000
            if domain:
                domain_phrase=self._phrase(domain)
                if domain_phrase and (domain_phrase in su or su in domain_phrase): score+=6500
                if domain in r['subject'].lower() or self._phrase(r['subject']) == domain_phrase: score += 7000
            if cphrase and cphrase in tp: score+=6500
            if tp and cphrase and tp in cphrase: score+=3500
            if cphrase and cphrase in sp: score+=3000
            if core_cphrase and core_cphrase==tp and core_cphrase!=cphrase: score+=26000
            if core_cphrase and core_cphrase==sp and core_cphrase!=cphrase: score+=9000
            # Small morphology-aware boost (message/messaging, normalize/normalization, etc.)
            # helps select the section that actually belongs to an exact concept.
            for cw in qset:
                if len(cw) >= 5:
                    for sw in section:
                        if len(sw) >= 6 and (sw.startswith(cw[:5]) or cw.startswith(sw[:5])):
                            score += 900
            # Token evidence with rarity weighting
            for w in qset:
                idf=self._idf(w)
                if w in topic: score += 1400*idf
                elif w in section: score += 800*idf
                elif w in subject: score += 500*idf
                elif w in definition: score += 120*idf
                elif w in syntax: score += 100*idf
            overlap=len(qset&topic)/len(qset) if qset else 0
            sec_overlap=len(qset&section)/len(qset) if qset else 0
            subj_overlap=len(qset&subject)/len(qset) if qset else 0
            sim=self._sim(cphrase,tp) if cphrase else 0
            score += 2600*overlap + 700*sec_overlap + 500*subj_overlap + 1500*sim
            if lang: score+=5000
            if domain: score+=4500
            # Exact section concept is strong evidence for subtopic questions.
            if cphrase and cphrase==sp: score+=5000
            # Exact section concept: prefer the first/overview topic in that section.
            if cphrase and cphrase==sp:
                score += max(0, 2200 - r['_section_rank']*180)
            # Subject/domain-only queries need a deterministic overview preference.
            low_topic=r['topic'].lower()
            if not cphrase or field_request:
                if r['_section_rank']==0: score += 900
                if low_topic:
                    if low_topic.startswith(('what is ','what are ','introduction to ')): score += 5000
                    if lang and low_topic == f'what is {lang}?': score += 10000
                    if domain and low_topic in {f'what is {domain}?', f'what is {domain}'}: score += 10000
                    if domain and (self._phrase(r['subject']) == self._phrase(domain) or self._phrase(r['subject']) in {self._phrase(a) for a in self.DOMAIN_ALIASES.get(domain,set())}): score += 9000
                    if domain and self._phrase(r['topic']) == self._phrase(domain): score += 12000
                    if domain in self.DOMAIN_OVERVIEW_HINTS:
                        hint_section, hint_topic = self.DOMAIN_OVERVIEW_HINTS[domain]
                        if hint_section and r['section'].lower() == hint_section: score += 11000
                        if hint_topic and self._phrase(r['topic']) == hint_topic: score += 15000
                    if domain == 'machine learning' and low_topic in ('what is machine learning?', 'what is machine learning'): score += 10000
                    if domain == 'artificial intelligence' and low_topic in ('what is artificial intelligence?', 'what is artificial intelligence'): score += 10000
                    if low_topic.startswith(('features of ','advantages','disadvantages','history of ')): score += 1800
            # Broad conceptual questions: choose overview/introduction over narrow variants.
            broad=(qtype in ('definition','explanation','normal') and len(qset)<=3)
            low=r['topic'].lower()
            if broad:
                if low.startswith(('what is ','what are ','introduction to ')): score+=2200
                if cphrase and cphrase==sp and not low.startswith(('what is ','what are ','introduction to ','artificial ')): score -= 16000
                if cphrase == 'neural network' and low.startswith('artificial neural network'): score += 6500
                if 'best practices' in low: score-=900
                if low.startswith(('advantages','disadvantages')): score-=700
                # If the query exactly names a section, prefer its introduction/overview.
                if cphrase and cphrase==sp and low.startswith(('introduction to ','what is ','what are ')): score+=2500
            # Strong anchor bonus to avoid cross-domain duplicates.
            if lang=='c': score+=2500
            results.append((score,i,overlap,sec_overlap,sim))
        results.sort(key=lambda x:(-x[0],-x[2],-x[3],-x[4],self.topic_index[x[1]]['topic'].lower()))
        # Confidence gate based on evidence, not raw score.
        top=results[0]; top_r=self.topic_index[top[1]]
        top_score=top[0]; second=results[1][0] if len(results)>1 else 0; margin=top_score-second
        evidence_overlap=top[2]; sec_overlap=top[3]; sim=top[4]
        anchored=bool(lang or domain)
        if cphrase and not field_request:
            # A rare/missing concept word must not be rescued by a common word.
            top_topic=set(top_r['_topic_tokens']); top_section=set(top_r['_section_tokens']);
            covered=sum(1 for w in qset if w in top_topic or w in top_section or w in top_r['_subject_tokens'])
            required=len(core_qset) if core_qset else len(qset)
            covered_core=sum(1 for w in core_qset if w in top_topic or w in top_section or w in top_r['_subject_tokens'])
            if anchored:
                if evidence_overlap<0.5 and sec_overlap<0.5 and sim<0.78: return []
                if required>=2 and covered_core < required: return []
            else:
                if evidence_overlap<0.5 and sec_overlap<0.5 and sim<0.80: return []
                if required>=2 and covered_core < required: return []
                if required>=2 and evidence_overlap<0.5 and sec_overlap<0.5 and margin<1000: return []
        else:
            # subject/domain-only queries should be accepted only with compatible corpus.
            if not anchored:return []
        # For domains without a genuine overview topic in the knowledge base,
        # do not manufacture an arbitrary answer from the first section.
        if not cphrase and domain in self.DOMAIN_OVERVIEW_HINTS:
            hint_section, hint_topic = self.DOMAIN_OVERVIEW_HINTS[domain]
            if not hint_section and not hint_topic:
                return []

        # Explicit ambiguity guard: if the exact same generic topic exists in
        # multiple subjects, do not arbitrarily select one without a domain/language
        # anchor. The caller can then ask the user to clarify the subject.
        if not anchored and full_phrase:
            exact_topic_ids=self.topic_phrase_index.get(full_phrase,set())
            if len(exact_topic_ids) > 1:
                exact_subjects={self.topic_index[i]['subject'] for i in exact_topic_ids}
                if len(exact_subjects) > 1:
                    return []
            if core_cphrase and core_cphrase != full_phrase:
                exact_core_ids=self.topic_phrase_index.get(core_cphrase,set())
                if len(exact_core_ids) > 1:
                    exact_core_subjects={self.topic_index[i]['subject'] for i in exact_core_ids}
                    if len(exact_core_subjects) > 1:
                        return []

        # If two top results are effectively tied but come from different subjects
        # and the query did not identify a domain/language, do not arbitrarily choose
        # one. Let the higher-level Gemini/fallback layer answer instead.
        if not anchored and len(results) > 1:
            a,b=results[0],results[1]
            ra,rb=self.topic_index[a[1]],self.topic_index[b[1]]
            same_topic = self._phrase(ra['topic']) == self._phrase(rb['topic']) and self._phrase(ra['topic']) == full_phrase
            if ra['subject'] != rb['subject'] and abs(a[0]-b[0]) <= max(250, abs(a[0])*0.05) and not same_topic and len(qset) <= 3:
                # Near-tied cross-subject matches are ambiguous even when the
                # stored definitions happen to be identical (e.g. DFS).
                return []

        out=[]
        for score,i,ov,sov,sim in results[:20]:
            r=dict(self.topic_index[i]); r.pop('_section_key',None); r.pop('_section_rank',None); r.pop('_subject_tokens',None); r.pop('_section_tokens',None); r.pop('_topic_tokens',None); r.pop('_definition_tokens',None); r.pop('_syntax_tokens',None); r.pop('_example_tokens',None); r.pop('_output_tokens',None); r.pop('_all_tokens',None)
            r['content']=self._text_raw({'subject':r['subject'],'section':r['section']},{'topic':r['topic'],'definition':r['definition'],'syntax':r['syntax'],'example':r['example'],'output':r['output'],'tips':r['tips']})
            r['_score']=round(score,2); r['_topic_overlap']=round(ov,4); r['_section_overlap']=round(sov,4); r['_fuzzy']=round(sim,4)
            out.append(r)
        self._domain_soft = False
        return out
