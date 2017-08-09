package pocketsphinx

/*
#cgo pkg-config: pocketsphinx sphinxbase
#include <pocketsphinx.h>
#include <err.h>
#include <stdio.h>
cmd_ln_t *default_config(){
    return cmd_ln_parse_r(NULL, ps_args(), 0, NULL, FALSE);
}
int process_raw(ps_decoder_t *ps, char const *data, size_t n_samples, int no_search, int full_utt){
    n_samples /= sizeof(int16);
    return ps_process_raw(ps, (int16 *)data, n_samples, no_search, full_utt);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

//Result is a speech recognition result
type Result struct {
	Text  string `json:"text"`
	Score int64  `json:"score"`
	Prob  int64  `json:"prob"`
}

//PocketSphinx is a speech recognition decoder object
type PocketSphinx struct {
	ps *C.ps_decoder_t
}

//NewPocketSphinx creates PocketSphinx instance with specific options.
func NewPocketSphinx(hmm string, dict string, samprate float64) *PocketSphinx {
	psConfig := C.default_config()

	path := C.CString("/dev/null")
	defer C.free(unsafe.Pointer(path))
	C.err_set_logfile(path)

	if hmm != "" {
		setStringParam(psConfig, "-hmm", hmm)
	}
	if dict != "" {
		setStringParam(psConfig, "-dict", dict)
	}
	setFloatParam(psConfig, "-samprate", samprate)

	ps := C.ps_init(psConfig)
	C.cmd_ln_free_r(psConfig)

	return &PocketSphinx{ps: ps}
}

//Free releases all resources associated with the PocketSphinx.
func (p *PocketSphinx) Free() {
	C.ps_free(p.ps)
}

//StartUtt starts utterance processing.
func (p *PocketSphinx) StartStream() error {
	ret := C.ps_start_stream(p.ps)
	if ret != 0 {
		return fmt.Errorf("start_stream error:%d", ret)
	}
	return nil
}

//StartUtt starts utterance processing.
func (p *PocketSphinx) StartUtt() error {
	ret := C.ps_start_utt(p.ps)
	if ret != 0 {
		return fmt.Errorf("start_utt error:%d", ret)
	}
	return nil
}

//EndUtt ends utterance processing.
func (p *PocketSphinx) EndUtt() error {
	ret := C.ps_end_utt(p.ps)
	if ret != 0 {
		return fmt.Errorf("end_utt error:%d", ret)
	}
	return nil
}

func bool2int(b bool) int {
	if b {
		return 1
	}
	return 0
}

//ProcessRaw processes a single channel, 16-bit pcm signal. if noSearch is true, ProcessRaw performs only feature extraction but don't do any recognition yet. if fullUtt is true, this block of data is a full utterance worth of data.
func (p *PocketSphinx) ProcessRaw(raw []int16, noSearch, fullUtt bool) error {
	raw_byte := (*C.char)(unsafe.Pointer(&raw[0]))
	numByte := len(raw) * 2
	processed := C.process_raw(p.ps, raw_byte, C.size_t(numByte), C.int(bool2int(noSearch)), C.int(bool2int(fullUtt)))
	if processed < 0 {
		return fmt.Errorf("process_raw error")
	}
	return nil
}

//GetHyp gets speech recognition result for best hypothesis.
func (p *PocketSphinx) GetHyp() (Result, error) {
	var score C.int32
	charp := C.ps_get_hyp(p.ps, &score)
	if charp == nil {
		return Result{}, errors.New("no hypothesis")
	}
	text := C.GoString(charp)
	ret := Result{Text: text, Score: int64(score), Prob: int64(C.ps_get_prob(p.ps))}
	return ret, nil
}

func (p *PocketSphinx) getNbestHyp(nbest *C.ps_nbest_t) Result {
	var score C.int32
	text := C.GoString(C.ps_nbest_hyp(nbest, &score))
	ret := Result{Text: text, Score: int64(score)}
	return ret
}

func (p *PocketSphinx) GetNbest(numNbest int) []Result {
	ret := make([]Result, 0, numNbest)

	nbestIt := C.ps_nbest(p.ps)
	for {
		if nbestIt == nil {
			break
		}
		if len(ret) == numNbest {
			C.ps_nbest_free(nbestIt)
			break
		}

		hyp := p.getNbestHyp(nbestIt)
		if hyp.Text == "" {
			C.ps_nbest_free(nbestIt)
			break
		}
		ret = append(ret, hyp)
		nbestIt = C.ps_nbest_next(nbestIt)
	}

	return ret
}

func (p *PocketSphinx) ProcessUtt(raw []int16, numNbest int) ([]Result, error) {
	ret := make([]Result, 0, numNbest)
	err := p.StartUtt()
	if err != nil {
		return ret, err
	}
	err = p.ProcessRaw(raw, false, true)
	if err != nil {
		return ret, err
	}
	err = p.EndUtt()
	if err != nil {
		return ret, err
	}
	r, err := p.GetHyp()
	if err == nil {
		ret = append(ret, r)
	} else {
		return ret, err
	}

	ret = append(ret, p.GetNbest(numNbest-1)...)
	return ret, nil
}

func (p *PocketSphinx) ParseJSGF(name string, grammar string) {
	cname := C.CString(name)
	cgrammar := C.CString(grammar)
	C.ps_set_jsgf_string(p.ps, cname, cgrammar)
	C.free(unsafe.Pointer(cname))
	C.free(unsafe.Pointer(cgrammar))
}

func (p *PocketSphinx) SetKeyphrase(name string, keyphrase string) {
	cname := C.CString(name)
	ckeyphrase := C.CString(keyphrase)
	C.ps_set_keyphrase(p.ps, cname, ckeyphrase)
	C.free(unsafe.Pointer(cname))
	C.free(unsafe.Pointer(ckeyphrase))
}

func (p *PocketSphinx) SetSearch(name string) {
	cname := C.CString(name)
	C.ps_set_search(p.ps, cname)
	C.free(unsafe.Pointer(cname))
}

func (p *PocketSphinx) GetSearch() string {
	cname := C.ps_get_search(p.ps)
	return C.GoString(cname)
}

func (p *PocketSphinx) IsInSpeech() bool {
	ret := C.ps_get_in_speech(p.ps)
	return ret == 1
}

func setStringParam(psConfig *C.cmd_ln_t, key, val string) {
	keyPtr := C.CString(key)
	defer C.free(unsafe.Pointer(keyPtr))
	valPtr := C.CString(val)
	defer C.free(unsafe.Pointer(valPtr))
	C.cmd_ln_set_str_r(psConfig, keyPtr, valPtr)
}

func setFloatParam(psConfig *C.cmd_ln_t, key string, val float64) {
	keyPtr := C.CString(key)
	defer C.free(unsafe.Pointer(keyPtr))
	C.cmd_ln_set_float_r(psConfig, keyPtr, C.double(val))
}

func setIntParam(psConfig *C.cmd_ln_t, key string, val int64) {
	keyPtr := C.CString(key)
	defer C.free(unsafe.Pointer(keyPtr))
	C.cmd_ln_set_int_r(psConfig, keyPtr, C.long(val))
}
